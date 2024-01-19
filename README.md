# MultiPing for Vultr

## Description

Used to ping Vultr servers for different regions before deploying.

![screenshot](assets/screenshot.png)

> - Left - Per IP ping result.
> - Right - Per region ping result.

## Usage


0. Install requirements
    ```bash
    pip install -r requirements.txt
    ```
1. Crawl Vultr server list
    ```bash
    make crawl
    ```
2. Start
    ```bash
    ./main.py
    ``````