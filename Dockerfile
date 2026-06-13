FROM python:3.11-slim

WORKDIR /app

# System deps (none needed beyond pip packages)
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    websocket-client

# Copy bot
COPY deriv_rise_fall_mc_v3.py .

# DATA_DIR lives on a Railway volume mounted at /data
# The bot reads this env var — set it in railway.toml / Railway dashboard
ENV DATA_DIR=/data

# Healthcheck — Railway requires a responsive healthcheck to avoid
# SIGTERM kills. The bot has no HTTP server, so we verify the process
# is alive by checking that the PID file exists (written by entrypoint).
# Timeout is generous because the historical collector can take ~30s.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD test -f /tmp/bot.pid && kill -0 $(cat /tmp/bot.pid) 2>/dev/null || exit 1

# Entrypoint writes PID then execs the bot
CMD sh -c 'echo $$ > /tmp/bot.pid && exec python -u deriv_rise_fall_mc_v3.py'
