import requests
from lxml.html import etree
import time
import json
import os
import re


class CrawlChelogo(object):
    def __init__(self):
        super(CrawlChelogo, self).__init__()
        self.page_urls = [
            "http://www.chelogo.com/chebiao/",
            "http://www.chelogo.com/chebiao/list_1_2.html",
            "http://www.chelogo.com/chebiao/list_1_3.html",
            "http://www.chelogo.com/chebiao/list_1_4.html",
            "http://www.chelogo.com/chebiao/list_1_5.html",
            "http://www.chelogo.com/chebiao/list_1_6.html",
        ]
        self.path = "./chelogo"

    def crawl(self):
        item_urls = []
        for page in self.page_urls:
            try:
                response = requests.get(page)
            except Exception as e:
                print(e)
                continue
            if response.status_code == 200:
                print(response.encoding)
                dom = etree.HTML(response.text)
                items = dom.xpath("""//*[@class="list_chebiao"]/ul""")
                for i in items:
                    a = i.xpath("""a""")[0]
                    url = a.get("href")
                    print(url)
                    item_urls.append(url)
            print("==================================")
            time.sleep(3)
        with open(os.path.join(self.path, "item_urls.json"), "w", encoding="utf-8") as f:
            json.dump(item_urls, f, ensure_ascii=False, indent=' ')

    def download_item(self):
        manual = []
        with open(os.path.join(self.path, "item_urls.json"), encoding="utf-8") as f:
            item_urls = json.load(f)
        for url in item_urls:
            try:
                response = requests.get(url)
            except Exception as e:
                manual.append(url)
                continue
            if response.status_code == 200:
                dom = etree.HTML(response.text)
                info_box = dom.xpath("""//div[@class="mainart"]""")[0]
                info_article = dom.xpath("""//div[@class="article_info"]""")[0]
                try:
                    text = self.process_info_article(info_article)
                    record, img = self.process_info_box(info_box)
                    record["简介"] = text
                    print(record)
                    print("==============================================")
                    self.save(record, img)
                except Exception as e:
                    manual.append(url)
            time.sleep(1)
        print("======== NEED MANUAL PROCESSING ========")
        print(manual)

    @staticmethod
    def process_info_box(info_box):
        """返回结构体，图片"""
        img = info_box.xpath("""div[@class="mainart_img"]/img""")[0]
        img_url = img.get("src")
        try:
            response = requests.get(img_url)
        except Exception as e:
            print(e)
            response = None
        if response.status_code == 200:
            image = response.content
        else:
            image = None
        box = info_box.xpath("""div[@class="mainart_info"]/li""")
        record = {
            "品牌": box[0].xpath("""a""")[0].get("title").encode("ISO-8859-1", errors="ignore").decode("gbk", errors="ignore"),
            "别名": box[1].text.encode("ISO-8859-1", errors="ignore").decode("gbk", errors="ignore").replace("别名：", ""),
            "产地": box[2].text.encode("ISO-8859-1", errors="ignore").decode("gbk", errors="ignore").replace("产地：", ""),
            "隶属公司": box[3].text.encode("ISO-8859-1", errors="ignore").decode("gbk", errors="ignore").replace("隶属公司：", ""),
            "成立时间": box[4].text.encode("ISO-8859-1", errors="ignore").decode("gbk", errors="ignore").replace("成立时间：", ""),
            "创始人": box[5].text.encode("ISO-8859-1", errors="ignore").decode("gbk", errors="ignore").replace("创始人：", ""),
            "官网": box[6].xpath("""a""")[0].get("href"),
        }
        return record, image

    @staticmethod
    def process_info_article(info_article):
        """获取正文内容，返回str"""
        text = ""
        try:
            paragraphs = info_article.xpath("""p""")
            text = [p.text.encode("ISO-8859-1", errors="ignore").decode("gbk", errors="ignore") for p in paragraphs]
            text = "\n".join([re.sub(r"\\u[0-9a-zA-Z]]{4}", "", p) for p in text])
        except Exception as e:
            print(e)
        finally:
            return text

    def save(self, record, image):
        dir = os.path.join(self.path, record["品牌"])
        os.makedirs(dir)
        with open(os.path.join(dir, "info.json"), "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=' ')
        with open(os.path.join(dir, "logo.jpg"), "wb") as f:
            f.write(image)


if __name__ == "__main__":
    crawler = CrawlChelogo()
    # crawler.crawl()
    crawler.download_item()







