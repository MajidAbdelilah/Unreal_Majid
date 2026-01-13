SRCS=  src/lib.rs  src/main.rs  src/core/entry_point.rs  src/core/mod.rs  src/core/renderer.rs	src/core/window.rs
CFLAGS= --release
CPP= cargo
NAME= particle-system

.PHONY: clean

all: ./target/release/$(NAME)

./target/release/$(NAME): $(SRCS)
	$(CPP) build $(CFLAGS)

clean:
	rm -rf target/release/build

fclean: clean
	rm -rf ./target/release/$(NAME)

re: fclean all
