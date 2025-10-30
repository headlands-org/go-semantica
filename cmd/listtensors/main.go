package main

import (
	"fmt"
	"log"
	"os"

	"github.com/lth/pure-go-llamas/internal/gguf"
)

func main() {
	if len(os.Args) < 2 {
		log.Fatalf("usage: %s <model.gguf>", os.Args[0])
	}
	path := os.Args[1]
	reader, err := gguf.Open(path)
	if err != nil {
		log.Fatalf("open gguf: %v", err)
	}
	defer reader.Close()

	for _, name := range reader.ListTensors() {
		fmt.Println(name)
	}
}
