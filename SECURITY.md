# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | Yes       |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT** open a public issue
2. Email the maintainer at **brolyroly007@gmail.com** with details
3. Include steps to reproduce the vulnerability
4. Allow up to 72 hours for an initial response

## Security Best Practices

When using this project:

- **Never commit `.env` files** - They may contain API keys and credentials
- **Rotate API keys regularly** - Especially if you suspect exposure
- **Use environment variables** - Never hardcode secrets in source code
- **Review proxy configurations** - Ensure proxies are from trusted sources
- **Keep dependencies updated** - Run `pip install --upgrade -r requirements.txt` regularly
