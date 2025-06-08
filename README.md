# Smart TV Streaming Controller

A Python application that automates streaming video site interactions and controls LG Smart TVs using computer vision pattern recognition.

## Features

- Automated browser control using Playwright
- Video frame analysis using OpenCV
- LG Smart TV control integration
- Support for headless and headful browser modes
- Concurrent processing of multiple channels
- SOLID principles implementation
- Asynchronous operation using asyncio
- Declarative browser flow configuration

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install Playwright browsers:
```bash
playwright install
```

3. Configure your settings in `config/config.yaml`

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your credentials
```

## Usage

Run the application:
```bash
python -m src.main
```

## Architecture

The application follows SOLID principles and is organized into the following modules:

- `browser/`: Browser automation and flow management
- `video/`: Video frame analysis and pattern detection
- `tv/`: LG Smart TV control integration
- `streaming/`: Streaming service management
- `utils/`: Utility functions and helpers

## Testing

Run tests:
```bash
pytest tests/
```

## License

MIT 