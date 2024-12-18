
"""
AI Dataset Creator - Main Entry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:author: 王宇 (Wang Yu)
:email: wywelljob@gmail.com
:copyright: (c) 2024 Wang Yu
:license: MIT, see LICENSE for more details.
"""

from ui.app import create_ui

if __name__ == "__main__":
    app = create_ui()
    app.launch(inbrowser=True)