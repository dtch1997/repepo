FROM python:3.10 AS main
ENV DEBIAN_FRONTEND=noninteractive
ENV GIT_URL="https://github.com/dtch1997/repepo"
LABEL org.opencontainers.image.source=${GIT_URL}

RUN apt-get update -q \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    # essential for running. GCC is for Torch triton
    git git-lfs build-essential \
    # essential for testing
    zip make \
    # devbox niceties
    curl vim tmux less sudo \
    # CircleCI
    ssh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Devbox niceties
WORKDIR "/devbox-niceties"
## the Unison file synchronizer
RUN curl -OL https://github.com/bcpierce00/unison/releases/download/v2.53.4/unison-2.53.4-ubuntu-x86_64-static.tar.gz \
    && tar xf unison-2.53.4-ubuntu-x86_64-static.tar.gz \
    && mv bin/unison bin/unison-fsmonitor /usr/local/bin/ \
    && rm -rf /devbox-niceties \
## Terminfo for the Alacritty terminal
    && curl -L https://raw.githubusercontent.com/alacritty/alacritty/master/extra/alacritty.info | tic -x /dev/stdin

# Create non-root user
ARG USERID=1001
ARG GROUPID=1001
ARG USERNAME=dev

# Simulate virtualenv activation
ENV VIRTUAL_ENV="/opt/venv"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN python3 -m venv "${VIRTUAL_ENV}" --system-site-packages \
    && addgroup --gid ${GROUPID} ${USERNAME} \
    && adduser --uid ${USERID} --gid ${GROUPID} --disabled-password --gecos '' ${USERNAME} \
    && usermod -aG sudo ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && mkdir -p "/workspace" \
    && chown -R ${USERNAME}:${USERNAME} "${VIRTUAL_ENV}" "/workspace"
USER ${USERNAME}
WORKDIR "/workspace"

COPY --chown=${USERNAME}:${USERNAME} . .
RUN pip install pdm \
    && pdm install --dev \
    && rm -rf "${HOME}/.cache"

# Set git remote URL to https
RUN git remote set-url origin "$(git remote get-url origin | sed 's|git@github.com:|https://github.com/|' )"

# Abort if repo is dirty
RUN echo "$(git status --porcelain --ignored=traditional | grep -v '.egg-info/$\|build/$')" \
    && if ! { [ -z "$(git status --porcelain --ignored=traditional | grep -v '.egg-info/$\|build/$')" ] \
    ; }; then exit 1; fi

# Default command to run -- may be changed at runtime
CMD ["/bin/bash"]
