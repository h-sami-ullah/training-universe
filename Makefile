PORT := ${PORT}
image_tag:=${image_tag}

AWS_ACCESS_KEY_ID := ${AWS_ACCESS_KEY_ID}
AWS_SECRET_ACCESS_KEY := ${AWS_SECRET_ACCESS_KEY}
all : build deploy
build: Dockerfile
	docker build --build-arg PORT=${PORT} -t ${image_tag} .
deploy:
	docker run -p ${PORT}:${PORT} -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} -e PORT=${PORT} ${image_tag}