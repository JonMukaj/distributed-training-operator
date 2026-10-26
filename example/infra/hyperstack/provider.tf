terraform {
  required_version = ">= 1.0"
  required_providers {
    hyperstack = {
      source  = "NexGenCloud/hyperstack"
      version = "1.56.0-alpha"
    }
  }
}

# Authentication is read from the HYPERSTACK_API_KEY environment variable:
#   export HYPERSTACK_API_KEY=your-api-key-here
# (the same key the operator's hyperstack provider uses). The provider block is
# intentionally empty — do not put the key in Terraform state or source.
provider "hyperstack" {}
