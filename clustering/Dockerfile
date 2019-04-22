FROM gzynda/tacc-maverick-ml:latest

LABEL maintainer "Josué Alfaro <josuejalfaro@gmail.com>"

ENV dataDir=/gnlp-project-clustering/pngs
COPY . /gnlp-project-clustering

WORKDIR /gnlp-project-clustering

CMD ["python", "./feature_extraction.py", ${dataDir}]
