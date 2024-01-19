.ONESHELL:

.PHONY: crawl

crawl:
	@cd crawler
	scrapy crawl info -L ERROR -O ../info.json