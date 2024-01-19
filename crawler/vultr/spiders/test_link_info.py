import scrapy
from scrapy.http import Response
from scrapy.shell import inspect_response


class TestLinkInfoSpider(scrapy.Spider):
    name = "info"
    allowed_domains = ["vultr.com"]
    start_urls = ["https://jnb-za-ping.vultr.com"]

    def parse(self, response: Response):
        """Extract candidate server infos."""
        for optgroup in response.css('optgroup'):
            optgroup: scrapy.Selector
            continent = optgroup.attrib['label']
            cities = (x.get() for x in optgroup.css('option::text'))
            for city, option in zip(cities, optgroup.css('option')):
                country_code = option.attrib['data-country'].upper()
                url = option.attrib['data-url']
                city_code = option.attrib['value'].upper()
                yield response.follow(
                    url=url,
                    callback=self.parse_url,
                    cb_kwargs={
                        'continent': continent,
                        'country_code': country_code,
                        'city': city,
                        'city_code': city_code,
                    },
                )

    def parse_url(self, response: Response, **kwargs):
        v4 = response.css('#useripv4::text')[0].get()
        v6 = response.css('#useripv6::text')[0].get()
        yield {'ipv4': v4, 'ipv6': v6, **kwargs}
