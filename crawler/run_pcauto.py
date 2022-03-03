import requests
from lxml.html import etree
import time
import json
import os


class CrawlPcauto(object):
    def __init__(self):
        super(CrawlPcauto, self).__init__()
        self.categories = [
            "https://www.pcauto.com.cn/zt/chebiao/guochan/",
            "https://www.pcauto.com.cn/zt/chebiao/riben/",
            "https://www.pcauto.com.cn/zt/chebiao/deguo/",
            "https://www.pcauto.com.cn/zt/chebiao/faguo/",
            "https://www.pcauto.com.cn/zt/chebiao/yidali/",
            "https://www.pcauto.com.cn/zt/chebiao/yingguo/",
            "https://www.pcauto.com.cn/zt/chebiao/meiguo/",
            "https://www.pcauto.com.cn/zt/chebiao/hanguo/",
            "https://www.pcauto.com.cn/zt/chebiao/qita/",
        ]
        self.path = "./pcauto"

    def crawl(self):
        name2url = {}
        for cat in self.categories:
            try:
                response = requests.get(cat)
            except Exception as e:
                print(e)
                continue
            if response.status_code == 200:
                print(response.encoding)
                dom = etree.HTML(response.text)
                blocks = dom.xpath("""//*[@id="pcauto"]/div[2]/div/div[2]/ul/li""")
                for b in blocks:
                    a = b.xpath("""div[@class="dTxt"]/i[@class="iTit"]/a""")
                    assert len(a) == 1
                    a = a[0]
                    url = "http://" + a.get("href").strip("/")
                    name = a.text.encode("ISO-8859-1").decode("gbk", errors="ignore")
                    print(name, url)
                    name2url[name] = url
            print("===============================")
            time.sleep(1)
        with open(os.path.join(self.path, "name2url.json"), "w", encoding="utf-8") as f:
            json.dump(name2url, f, ensure_ascii=False, indent=' ')

    def download_image(self):
        with open(os.path.join(self.path, "name2url.json"), encoding="utf-8") as f:
            name2url = json.load(f)
        for name, url in name2url.items():
            try:
                response = requests.get(url)
            except Exception as e:
                print(e)
                continue
            if response.status_code == 200:
                dom = etree.HTML(response.text)
                img = dom.xpath("""//*[@id="article"]//img""")
                assert len(img) == 1
                img = img[0]
                img_url = "http://" + img.get("src").strip("/")
                try:
                    image = requests.get(img_url)
                except Exception as e:
                    print(e)
                    continue
                with open(os.path.join(self.path, "{}.jpg".format(name)), "wb") as f:
                    f.write(image.content)
                print("{}.jpg saved".format(name))
            time.sleep(0.5)


if __name__ == "__main__":
    crawler = CrawlPcauto()
    # crawler.crawl()
    crawler.download_image()

