import bz2
import os
import xml.etree.ElementTree as ET
import json

dump_file = "wikiarticles.xml.bz2"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

'''
This file is used to extract Wikipedia articles from a compressed XML dump file.
It processes the dump, extracts article titles and text, and saves them in JSON format.
'''

PAGES_PER_FILE = 10000 # number of pages per output JSON file
MAX_PAGES = None # testing purposes to not parse full dump, set to None to parse all

page_count = 0
file_count = 1
pages_in_file = []

def write_json_file(pages, file_idx):
    filename = os.path.join(output_dir, f"wiki_{file_idx:04d}.json")
    with open(filename, "w", encoding="utf-8") as f:
        for page in pages:
            json.dump(page, f, ensure_ascii=False)
            f.write("\n")
    print(f"Written {len(pages)} pages to {filename}")

page_buffer = []
inside_page = False

with bz2.open(dump_file, "rt", encoding="utf-8") as f:
    for line in f:
        if "<page>" in line:
            inside_page = True
            page_buffer = [line]
        elif "</page>" in line:
            page_buffer.append(line)
            inside_page = False
            page_count += 1
            try:
                root = ET.fromstring("".join(page_buffer))
                title = root.findtext("title")
                text_elem = root.find(".//text")
                
                if text_elem is not None and text_elem.text and text_elem.text.strip():
                    text_content = text_elem.text.strip()

                    # Skip redirect pages
                    if text_content.lower().startswith("#redirect"):
                        continue

                    page_dict = {
                        "title": title,
                        "text": text_content.replace("\n", " ")
                    }
                    pages_in_file.append(page_dict)

                    # Write JSON file when limit reached
                    if len(pages_in_file) >= PAGES_PER_FILE:
                        write_json_file(pages_in_file, file_count)
                        pages_in_file = []
                        file_count += 1

                if MAX_PAGES and page_count >= MAX_PAGES:
                    break

            except ET.ParseError:
                continue
        elif inside_page:
            page_buffer.append(line)

# Write remaining pages
if pages_in_file:
    write_json_file(pages_in_file, file_count)

print(f"Extraction complete! Total pages processed: {page_count}")
