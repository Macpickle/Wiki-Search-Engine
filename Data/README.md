# Wiki Dump Parser

This project helps you extract Wikipedia articles from a compressed XML dump file. It processes the Wikipedia dump, extracts article titles and text, and saves them in JSON format. You can use this to convert the XML dump into structured JSON data for further processing.

## How to Use

### Step 1: Download Wikipedia Dump

1. Go to the [Wikipedia Dumps page](https://dumps.wikimedia.org/) and download the latest `wikiarticles.xml.bz2` file (or any other XML dump, but it has not been tested on another dump).

2. Once you have downloaded the dump, **upload it** to the `Data/` folder in this repository. Make sure the file is named `wikiarticles.xml.bz2` for the script to work correctly.

### Step 2: Run the Parser Script

After placing the `wikiarticles.xml.bz2` file in the `Data/` folder, you can run the `WikipediaParse.py` script to parse the compressed XML dump.

1. Make sure you have the necessary dependencies installed. If you're using a Python virtual environment, activate it first. If you don’t have one, create it using the following commands:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows, use .venv\Scripts\activate
   ```

2. Install the required Python libraries (if you haven't already):
    ```bash
    pip install -r requirements.txt
    ```

3. Now, run the WikipediaParse.py script:
    ```bash
    python WikipediaParse.py
    ```

This script will read the `wikiarticles.xml.bz2` dump file from the `Data/` folder, parse it, and save the extracted Wikipedia articles as JSON files in the `output/` folder.

The cleaning of the text from the page is crutial to the quality of the search/embedding. The current file that handles the cleaning of the wikipedia dump data is very rough, if you would like to improve the quality of the cleaning of text, you can, but may take a much longer time to parse.

If you wish to use a different dump file, simply open the **`WikipediaParse.py`** script and modify the line:
```python
dump_file = "wikiarticles.xml.bz2"
```

### Step 3: View the Output
Once the script finishes running, it will generate JSON files containing the parsed Wikipedia article titles and text. The files will be saved in the output/ folder. Each JSON file will contain up to 10,000 pages (this can be adjusted by modifying the PAGES_PER_FILE variable in the script).

### Customizing the Script
**PAGES_PER_FILE**: This controls how many pages are written to each JSON file. By default, it's set to 10,000. You can change this value if needed.

**MAX_PAGES**: This allows you to limit the number of pages to parse. Set it to None to parse the entire dump. It’s useful for testing purposes.

Example Output Format
Each JSON file will contain **PAGES_PER_FILE** number of json statements, each of which are specific articles.

```json
{
  "title": "Example Title",
  "text": "This is the content of the Wikipedia article."
}
```
Notes
This script will skip redirect pages (pages that start with #redirect).

If you encounter any issues, make sure your XML dump is valid and that it's the correct format for the script.


### Directory Structure
```
/WikiSearchEngine
│
└── /Data
  ├──/output                   # Output folder for the parsed JSON files, will be created automatically
  ├── README.md                # This file
  ├── wikiarticles.xml.bz2     # Place your downloaded Wikipedia dump here
  └──WikipediaParse.py         # Python script to parse the Wikipedia dump
```