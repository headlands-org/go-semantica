package main

import (
	"fmt"
	"log"
	"os"

	"github.com/lth/pure-go-llamas/internal/runtime"
)

func main() {
	if len(os.Args) < 3 {
		log.Fatalf("usage: %s <model.gguf> <text>", os.Args[0])
	}
	model, err := runtime.LoadModel(os.Args[1])
	if err != nil {
		log.Fatalf("load model: %v", err)
	}
	defer model.Close()

	toks, err := model.Tokenizer().Encode(os.Args[2])
	if err != nil {
		log.Fatalf("encode: %v", err)
	}
	fmt.Println(toks)
}
