# Web Content Extractor, Summarizer, and Translator

This Python project is designed to extract web page content, generate summaries, translate the content to Turkish, and store the data in a vector database.

## Features

1. **Web Content Extraction**: The program uses the DuckDuckGo search engine to search for web pages based on a given query, and then extracts the content of the top search results.

2. **Text Summarization**: The extracted web page content is summarized using the Ollama language model.

3. **Translation to Turkish**: The summaries are translated to Turkish using the Ollama language model.

4. **Vector Database Storage**: The extracted content, summaries, and translated summaries are stored in a Chroma vector database.

5. **Local File Storage**: The extracted content, summaries, and translated summaries are also saved to local text files.

## Requirements

- Python 3.7 or higher
- The following Python packages:
  - `duckduckgo-search`
  - `requests`
  - `beautifulsoup4`
  - `chromadb`
  - `langchain`
  - `sentence-transformers`
  - `numpy`

## Usage

1. Install the required packages using pip:

   ```
   pip install -r requirements.txt
   ```

2. Set the `OLLAMA_ENDPOINT` variable in the `main.py` file to the appropriate URL for your Ollama language model instance.

3. Run the `main.py` script:

   ```
   python main.py
   ```

   The program will perform the following actions:

   - Search for web pages using the `QUERY` variable
   - Extract the content of the top search results
   - Summarize the content using the Ollama language model
   - Translate the summaries to Turkish using the Ollama language model
   - Store the content, summaries, and translated summaries in a Chroma vector database
   - Save the content, summaries, and translated summaries to local text files

## Contributing

If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).