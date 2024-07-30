xhost +local:docker

docker run -itd \
    -v `pwd`/..:/root/robustness_ws \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY  \
    --rm \
    --name basalt basalt:armadillo bash

docker exec -it basalt bash

docker stop basalt

# xhost -local:docker
