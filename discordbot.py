import os
import discord
from dotenv import load_dotenv
import requests

load_dotenv()

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
ALLOWED_GUILD_ID = int(os.getenv("ALLOWED_GUILD_ID"))


@client.event
async def on_ready():
    print(f"We have logged in as {client.user}")


@client.event
async def on_message(message):
    if ((message.guild is None) or (message.guild.id != ALLOWED_GUILD_ID)):
        return

    if message.author == client.user:
        return

    if message.content.lower().startswith("$promp"):
        parts = message.content.split(maxsplit=1)
        prompt = parts[1] if len(parts) > 1 else None
        if prompt is None:
            await message.channel.send("Please provide a prompt after $promp.")
            return

        url = "http://localhost:5000/"
        params = {"prompt": prompt}
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.text
            await message.channel.send(data)
        except requests.exceptions.RequestException as e:
            await message.channel.send(f"An error occurred: {e}")


client.run(os.getenv("DISCORD_BOT_TOKEN"))