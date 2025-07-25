# Pixels RAG

## Usage

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Set up the environment variables. Copy-paste the `.env.example` file to `.env` and fill in the required values:
   1. If you plan to use the Discord bot, fill in the `DISCORD_BOT_TOKEN` and `ALLOWED_GUILD_ID` values in the `.env` file.
   2. If you want to use OpenAI's models, just fill the `OPENAI_API_KEY` value. All the other variables are already set to their default values.
   3. If you want to use a local model, adjust the `MODEL` and `EMBEDDING_MODEL` variables accordingly. A list of small models can be found in the `models_list.txt` file.
   Example:
   ```
   MODEL=llmware/bling-phi-3
   EMBEDDING_MODEL=intfloat/e5-small
   ```
3. Place your Pixels backup(s) in the `diary` folder.
4. Run the script to generate the RAG database:
```bash
python generate_embeddings.py
```
5. This will create a ChromaDB database in the `db` folder, which will be used for querying the data. You will be prompted to enter a profile name for the database if it doesn't already exist. This profile name will be used to identify your database in future queries. If you have multiple Pixels backups, you can create multiple profiles by running the script again with a different profile name.
6. To delete a profile, enter "del" when prompted for a profile name. You will then be asked to enter the profile name you want to delete. If the profile exists, it will be deleted from the database.

Next, you can either run the models in the **command line interface**, in a **Flask server**, or through a **Discord bot**:

## Command line interface
1. Run the CLI script:
```bash
python main.py
```
2. Enter your query when prompted.

## Running the Flask server
1. Start the Flask server:
```bash
python server.py
```
2. Send a POST request to the Flask server with your query:
```bash
POST http://localhost:8000/?prompt=Your prompt here
```


## Discord Bot
1. Set up a Discord bot and invite it to your server.
2. Run the Discord bot script:
```bash
python discordbot.py
```

3. Use the command `$prompt Your prompt here` in your Discord server to get a response from the bot.

## TODO
- [ ] Add memory to continue conversations.
- [ ] Add a web interface for the Flask server.