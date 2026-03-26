import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import urlsplit, urlunsplit

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/rss+xml, application/xml, text/xml;q=0.9, */*;q=0.8",
}

def clean_url(url):
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))

def extract_reddit_external(url):
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        tag = soup.find("shreddit-post")
        if tag and tag.get("content-href"):
            link = tag.get("content-href")
            if "reddit.com" not in link and "redd.it" not in link:
                return clean_url(link)
        body = soup.find("div", {"data-testid": "post-content"})
        if body:
            for a in body.find_all("a", href=True):
                href = a["href"]
                if href.startswith("http") and "reddit.com" not in href and "redd.it" not in href and "i.redd.it" not in href and "v.redd.it" not in href and "reddit.com/gallery" not in href:
                    return clean_url(href)
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("http") and "reddit.com" not in href and "redd.it" not in href and "reddit.com/gallery" not in href:
                return clean_url(href)
    except Exception:
        pass
    return clean_url(url)

def _download_feed_content(rss_url):
    try:
        response = requests.get(rss_url, headers=DEFAULT_HEADERS, timeout=10, allow_redirects=True)
        if response.ok and response.content:
            return response.content
    except Exception:
        pass
    return None

def fetch_rss_headlines(rss_url, limit=50):
    feed_content = _download_feed_content(rss_url)
    url_feed = feedparser.parse(rss_url)
    content_feed = feedparser.parse(feed_content) if feed_content else None
    feed = url_feed
    if content_feed and len(content_feed.entries) > len(url_feed.entries):
        feed = content_feed
    articles = []
    for entry in feed.entries[:limit]:
        title = entry.get("title", "").strip()
        rss_link = entry.get("link", "")
        if not title or not rss_link:
            continue
        published = None
        if "published_parsed" in entry and entry.published_parsed:
            published = datetime(*entry.published_parsed[:6])
        elif "updated_parsed" in entry and entry.updated_parsed:
            published = datetime(*entry.updated_parsed[:6])
        original_url = clean_url(rss_link)
        if "reddit.com" in rss_link:
            original_url = extract_reddit_external(rss_link)
        if original_url.startswith("https://i.redd.it") or original_url.startswith("https://v.redd.it"):
            continue
        if "/r/" in original_url:
            continue
        articles.append({"title": title, "rss_link": rss_link, "original_url": original_url, "published": published})
    return articles
