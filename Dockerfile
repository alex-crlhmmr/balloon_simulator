FROM python:3.10-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      libhdf5-dev \
      libnetcdf-dev \
      gfortran \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/output
VOLUME /app/output

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir netCDF4 pydap


COPY propagate.py weather.py geo_utils.py gas_utils.py constants.py ./

ENTRYPOINT ["python", "propagate.py"]
CMD [ \
  "--initial-lat","36.4", \
  "--initial-lon","-123.9", \
  "--initial-height","100.0", \
  "--num-simulations","1", \
  "--duration-hours","1", \
  "--num-cpus","7" \
]
