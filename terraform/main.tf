terraform {
  required_version = ">= 1.5"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  # Store Terraform state in GCS — create the bucket manually first:
  # gsutil mb -p PROJECT_ID gs://PROJECT_ID-tf-state
  backend "gcs" {
    bucket = "production-rag-system-tf-state"
    prefix = "rag-api"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# ── Enable required GCP APIs ──────────────────────────────────────────────────

resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "sqladmin.googleapis.com",
    "redis.googleapis.com",
    "secretmanager.googleapis.com",
    "artifactregistry.googleapis.com",
    "vpcaccess.googleapis.com",
    "servicenetworking.googleapis.com",
  ])

  service            = each.key
  disable_on_destroy = false
}

# ── VPC + Serverless VPC Connector ───────────────────────────────────────────

resource "google_compute_network" "vpc" {
  name                    = "rag-vpc"
  auto_create_subnetworks = false
  depends_on              = [google_project_service.apis]
}

resource "google_compute_subnetwork" "subnet" {
  name          = "rag-subnet"
  ip_cidr_range = "10.0.0.0/24"
  region        = var.region
  network       = google_compute_network.vpc.id
}

resource "google_vpc_access_connector" "connector" {
  name          = "rag-connector"
  region        = var.region
  ip_cidr_range = "10.8.0.0/28"
  network       = google_compute_network.vpc.name
  depends_on    = [google_project_service.apis]
}

# Private services access for Cloud SQL
resource "google_compute_global_address" "private_ip" {
  name          = "rag-private-ip"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.vpc.id
}

resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = google_compute_network.vpc.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip.name]
}

# ── Artifact Registry ─────────────────────────────────────────────────────────

resource "google_artifact_registry_repository" "repo" {
  location      = var.region
  repository_id = "rag-api"
  description   = "Docker images for RAG API"
  format        = "DOCKER"
  depends_on    = [google_project_service.apis]
}

# ── Cloud SQL (PostgreSQL 16 + pgvector) ──────────────────────────────────────

resource "google_sql_database_instance" "postgres" {
  name             = "rag-postgres"
  database_version = "POSTGRES_16"
  region           = var.region

  settings {
    tier              = "db-f1-micro"
    availability_type = "ZONAL"
    disk_size         = 10
    disk_autoresize   = true

    ip_configuration {
      ipv4_enabled                                  = false
      private_network                               = google_compute_network.vpc.id
      enable_private_path_for_google_cloud_services = true
    }

    backup_configuration {
      enabled    = true
      start_time = "03:00"
    }
  }

  deletion_protection = false
  depends_on          = [google_service_networking_connection.private_vpc_connection]
}

resource "google_sql_database" "ragdb" {
  name     = "ragdb"
  instance = google_sql_database_instance.postgres.name
}

resource "google_sql_user" "raguser" {
  name     = "raguser"
  instance = google_sql_database_instance.postgres.name
  password = var.db_password
}

# ── Memorystore Redis ─────────────────────────────────────────────────────────

resource "google_redis_instance" "cache" {
  name           = "rag-redis"
  tier           = "BASIC"
  memory_size_gb = 1
  region         = var.region

  authorized_network = google_compute_network.vpc.id
  connect_mode       = "PRIVATE_SERVICE_ACCESS"

  redis_version = "REDIS_7_0"
  display_name  = "RAG API Redis Cache"

  depends_on = [google_service_networking_connection.private_vpc_connection]
}

# ── Secret Manager ────────────────────────────────────────────────────────────

resource "google_secret_manager_secret" "db_password" {
  secret_id  = "rag-db-password"
  depends_on = [google_project_service.apis]

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "db_password" {
  secret      = google_secret_manager_secret.db_password.id
  secret_data = var.db_password
}

resource "google_secret_manager_secret" "openai_api_key" {
  secret_id  = "rag-openai-api-key"
  depends_on = [google_project_service.apis]

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "openai_api_key" {
  secret      = google_secret_manager_secret.openai_api_key.id
  secret_data = var.openai_api_key
}

resource "google_secret_manager_secret" "langsmith_api_key" {
  secret_id  = "rag-langsmith-api-key"
  depends_on = [google_project_service.apis]

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "langsmith_api_key" {
  secret      = google_secret_manager_secret.langsmith_api_key.id
  secret_data = var.langsmith_api_key
}

# ── Service Account for Cloud Run ─────────────────────────────────────────────

resource "google_service_account" "cloud_run_sa" {
  account_id   = "rag-api-sa"
  display_name = "RAG API Cloud Run Service Account"
}

resource "google_project_iam_member" "cloud_run_sa_roles" {
  for_each = toset([
    "roles/secretmanager.secretAccessor",
    "roles/cloudsql.client",
    "roles/run.invoker",
  ])

  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

# ── Cloud Run Service ─────────────────────────────────────────────────────────

resource "google_cloud_run_v2_service" "api" {
  name     = "rag-api"
  location = var.region

  template {
    service_account = google_service_account.cloud_run_sa.email

    vpc_access {
      connector = google_vpc_access_connector.connector.id
      egress    = "PRIVATE_RANGES_ONLY"
    }

    scaling {
      min_instance_count = 0
      max_instance_count = 3
    }

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/rag-api/rag-api:latest"

      resources {
        limits = {
          cpu    = "2"
          memory = "4Gi"
        }
        cpu_idle          = true
        startup_cpu_boost = true
      }

      # Application config
      env {
        name  = "APP_ENV"
        value = "production"
      }
      env {
        name  = "LOG_LEVEL"
        value = "INFO"
      }
      env {
        name  = "EMBEDDING_MODEL"
        value = "BAAI/bge-large-en-v1.5"
      }
      env {
        name  = "EMBEDDING_DIMENSION"
        value = "1024"
      }

      # PostgreSQL (private IP from Cloud SQL)
      env {
        name  = "POSTGRES_HOST"
        value = google_sql_database_instance.postgres.private_ip_address
      }
      env {
        name  = "POSTGRES_PORT"
        value = "5432"
      }
      env {
        name  = "POSTGRES_USER"
        value = "raguser"
      }
      env {
        name  = "POSTGRES_DB"
        value = "ragdb"
      }
      env {
        name = "POSTGRES_PASSWORD"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.db_password.secret_id
            version = "latest"
          }
        }
      }

      # Redis (Memorystore private IP)
      env {
        name  = "REDIS_HOST"
        value = google_redis_instance.cache.host
      }
      env {
        name  = "REDIS_PORT"
        value = "6379"
      }

      # OpenAI
      env {
        name = "OPENAI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.openai_api_key.secret_id
            version = "latest"
          }
        }
      }

      # LangSmith
      env {
        name = "LANGSMITH_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.langsmith_api_key.secret_id
            version = "latest"
          }
        }
      }
      env {
        name  = "LANGSMITH_PROJECT"
        value = "rag-api"
      }

      startup_probe {
        http_get {
          path = "/health"
          port = 8000
        }
        initial_delay_seconds = 30
        period_seconds        = 10
        failure_threshold     = 10
      }

      liveness_probe {
        http_get {
          path = "/health"
          port = 8000
        }
        period_seconds    = 30
        failure_threshold = 3
      }

      ports {
        container_port = 8000
      }
    }
  }

  depends_on = [
    google_project_service.apis,
    google_vpc_access_connector.connector,
  ]
}

# Allow unauthenticated access (public API)
resource "google_cloud_run_v2_service_iam_member" "public" {
  project  = google_cloud_run_v2_service.api.project
  location = google_cloud_run_v2_service.api.location
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
