# Heroku Deployment

Option 1: Procfile (dyno)
- heroku create
- heroku stack:set container
- heroku config:set APP_ENV=prod
- git add . && git commit -m "deploy" && git push heroku main

Option 2: Docker
- heroku container:login
- heroku create <app-name>
- heroku container:push web -a <app-name>
- heroku container:release web -a <app-name>

Scale dynos for real-time performance:
- heroku ps:scale web=1

Ensure logging and metrics via Heroku dashboard.
