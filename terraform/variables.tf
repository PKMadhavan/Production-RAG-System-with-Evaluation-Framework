variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "db_password" {
  description = "PostgreSQL raguser password"
  type        = string
  sensitive   = true
}

variable "openai_api_key" {
  description = "OpenAI API key for RAGAS evaluation"
  type        = string
  sensitive   = true
}

variable "langsmith_api_key" {
  description = "LangSmith API key for observability"
  type        = string
  sensitive   = true
}
