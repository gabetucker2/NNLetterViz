from nicegui import ui
import asyncio

count = 0
max_buttons = 10

async def add_buttons_periodically():
    global count
    while count < max_buttons:
        with ui.row():
            ui.button(f'Button {count}', on_click=lambda c=count: print(f'Clicked {c}'))
        count += 1
        await asyncio.sleep(3)

ui.page("/")(lambda: None)  # Ensures the page loads before the coroutine starts
ui.run(on_startup=add_buttons_periodically)
