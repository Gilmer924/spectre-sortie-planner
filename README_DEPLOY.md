# SPECTRE Sortie Planner — Deployment Guide

## Overview

This document provides basic steps for deploying the SPECTRE Sortie Planner web app using Docker. The application is written in Python and uses Streamlit for its web interface.

---

## 1. Prerequisites

- **Docker** installed on the server ([Get Docker](https://docs.docker.com/get-docker/))
- Access to this repository (via GitHub, GitLab, or download)

---

## 2. Building the Docker Image

1. **Clone or download** the SPECTRE repository to your local machine or server:

   ```sh
   git clone https://github.com/Gilmer924/spectre-sortie-planner.git
   cd SPECTRE_Sortie_Planner_Web

## 3. Quick Start
docker build -t spectre-sortie-planner . && docker run -p 8501:8501 spectre-sortie-planner
