from main import infer_with_model, DB_PROFILE
from dotenv import load_dotenv
import discord
import logging
import os


load_dotenv()
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
ALLOWED_GUILD_ID = int(os.getenv("ALLOWED_GUILD_ID"))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)




@client.event
async def on_ready():
    logger.info(f"Bot connected as {client.user}")


@client.event
async def on_message(message):
    if (message.guild is None) or (message.guild.id != ALLOWED_GUILD_ID) or (message.author == client.user):
        return

    if message.content.lower().startswith("$promp"):
        parts = message.content.split(maxsplit=1)
        if len(parts) < 2:
            await message.channel.send("Usage: `$promp <your prompt>`")
            return

        prompt = parts[1]
        await message.channel.send("Processing your prompt...")

        try:
            response = infer_with_model(prompt)
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            await message.channel.send("An error occurred during inference.")
            return

        await message.channel.send(response)

        os.makedirs("./responses", exist_ok=True)
        with open(f"./responses/{DB_PROFILE}.txt", "a", encoding="utf-8") as f:
            f.write(f"Prompt: {prompt}\nResponse: {response}\n\n")




client.run(DISCORD_BOT_TOKEN)