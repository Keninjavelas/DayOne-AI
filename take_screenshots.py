import asyncio
from playwright.async_api import async_playwright
import time
import os

async def main():
    print("Starting playwright...")
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        print("Navigating to http://localhost:8502...")
        await page.goto("http://localhost:8502")
        
        print("Waiting for login card...")
        await page.wait_for_selector(".login-card", timeout=15000)
        await asyncio.sleep(3)
        
        os.makedirs("assets", exist_ok=True)
        print("Taking login screenshot...")
        await page.screenshot(path="assets/login_screen.png", full_page=True)
        
        print("Logging in...")
        await page.fill('input[type="text"]', 'testuser')
        await page.fill('input[type="password"]', 'password123')
        await page.click('div[data-testid="stFormSubmitButton"] button')
        
        print("Waiting for zero state...")
        await page.wait_for_selector("text=Good Morning", timeout=15000)
        await asyncio.sleep(4)
        
        print("Taking logged in screenshot...")
        await page.screenshot(path="assets/command_center.png", full_page=True)
        # Send a prompt to trigger the chat UI
        print("Sending chat message...")
        await page.fill('textarea[aria-label="Ask DayOne anything... PTO, benefits, expenses"]', "What is our PTO policy?")
        await page.keyboard.press("Enter")
        
        print("Waiting for response...")
        await page.wait_for_selector(".stChatMessage", timeout=20000)
        await asyncio.sleep(5)
        
        print("Taking chat screenshot...")
        await page.screenshot(path="assets/chat_interface.png", full_page=True)
        
        await browser.close()
        print("Done!")

asyncio.run(main())
