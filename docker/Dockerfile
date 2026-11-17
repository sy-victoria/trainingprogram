FROM python:3.11.11-slim-bookworm
ARG EVALPLUS_REV=26d6d00bb1fd0fa37f39c99d5290da67891d1c5e
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# Only import EvalPlus's official execution primitives, not its model-generation stack.
RUN pip install --no-cache-dir numpy==2.2.6 psutil==7.2.2 setuptools==80.9.0 setuptools-scm==8.3.1 wget==3.2 appdirs==1.4.4 tempdir==0.7.1 datasets==5.2.0 \
    && git clone https://github.com/evalplus/evalplus.git /opt/evalplus \
    && git -C /opt/evalplus checkout "$EVALPLUS_REV" \
    && pip install --no-cache-dir --no-deps --no-build-isolation /opt/evalplus \
    && pip freeze > /opt/requirements-resolved.txt
COPY docker/judge_worker.py /opt/judge_worker.py
ENV PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
USER 65534:65534
WORKDIR /tmp
