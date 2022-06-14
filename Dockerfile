FROM python:3.8

RUN apt-get update
RUN apt-get -y install locales && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm
ENV HOME="/myapp"

RUN apt-get install -y vim less wget xz-utils
RUN apt-get install -y libgl1-mesa-dev

RUN mkdir -p ${HOME}
WORKDIR ${HOME}
COPY . ${HOME}

RUN pip install -r requirements.txt

# CMD ["python","main.py","-cam"]