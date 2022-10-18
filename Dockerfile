FROM python:3
COPY . /usr/app
EXPOSE 5000
WORKDIR /usr/app
RUN make install
CMD python3 app.py
