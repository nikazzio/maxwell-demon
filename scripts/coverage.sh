#!/usr/bin/env bash
set -euo pipefail

pytest --cov=maxwell_demon --cov-branch --cov-report=xml --cov-report=term-missing
