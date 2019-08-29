---
layout: posts
author: Huan
title: Django & Postgres with Docker in Travis CI
---

Connecting Django & Postgres with Docker in Travis CI.

---

Code on my [Github](https://github.com/ChuaCheowHuan/web_app)

---

If you see the following error in the Travis's job log while attempting to test
dockerized Django apps with Travis, it means that the postgres container has
started but not yet ready to accept connections.

```
psycopg2.OperationalError: could not connect to server: Connection refused
539	Is the server running on host "db" (172.18.0.2) and accepting
540	TCP/IP connections on port 5432?

.
.
.

django.db.utils.OperationalError: could not connect to server: Connection
refused 587	Is the server running on host "db" (172.18.0.2) and accepting
588	TCP/IP connections on port 5432?

The command "docker-compose run web python manage.py test" exited with 1.
```

A solution for this issue is to introduce a delay until connection is ready
before executing the test.

The delay has to be implemented in the ```docker-compose.yml``` file before
migration & running of Django's server.

```
command: bash -c 'while !</dev/tcp/db/5432; do sleep 1; done; python3 manage.py migrate'
```

```
command: bash -c 'while !</dev/tcp/db/5432; do sleep 1; done; python3 manage.py runserver 0.0.0.0:8000'
```

---

**Config files:**

These are the relevant config files used in a Django project with the delay
introduced in the ```docker-compose.yml``` file. The actual command to run the
test is in the ```.travis.yml``` file.

The database configuration in ```settings.py```
```
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'postgres',
        'USER': 'postgres',
        'HOST': 'db',
        'PORT': 5432,
        #'PORT': 5433,
    }
}
```

The ```Dockerfile```:
```
FROM python:3
WORKDIR /usr/src/app
ADD requirements.txt /usr/src/app
RUN pip install -r requirements.txt
ADD . /usr/src/app
```

The ```docker-compose.yml``` file:
```
version: '3'

services:
    db:
        image: postgres
    migration:
        build: .
#        command: python3 manage.py migrate
        command: bash -c 'while !</dev/tcp/db/5432; do sleep 1; done; python3 manage.py migrate'
        volumes:
            - .:/usr/src/app
        depends_on:
            - db
    web:
        build: .
#        command: python3 manage.py runserver 0.0.0.0:8000
        command: bash -c 'while !</dev/tcp/db/5432; do sleep 1; done; python3 manage.py runserver 0.0.0.0:8000'
        volumes:
            - .:/usr/src/app
        ports:
            - "8000:8000"
        depends_on:
            - db
            - migration
```

The ```.travis.yml``` file:
```
language: python
python:
    - 3.6
services:
    - docker
#    - postgres
install:
    - pip install -r requirements.txt
#before_script:
#    - psql -c 'create database testdb;' -U postgres
#    - psql -c 'create database travisci;' -U postgres
script:
#    - docker-compose build
#    - docker-compose run web python manage.py migrate
    - docker-compose run web python manage.py test
#    - python manage.py test
```

---

After introducing the delay, this is the successful test result output in
Travis's job log.

```
.
.
.
.......
528----------------------------------------------------------------------
529Ran 10 tests in 0.126s
530
531OK
532Destroying test database for alias 'default'...
533The command "docker-compose run web python manage.py test" exited with 0.
```

---

## References:

[stackoverflow](https://stackoverflow.com/questions/35069027/docker-wait-for-postgresql-to-be-running)

---

<br>
