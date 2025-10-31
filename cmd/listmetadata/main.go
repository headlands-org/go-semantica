package main

import (
	"fmt"
	"log"
	"os"
	"reflect"
	"unsafe"

	"github.com/headlands-org/go-semantica/internal/gguf"
)

func main() {
	if len(os.Args) != 2 {
		log.Fatalf("usage: %s <model.gguf>", os.Args[0])
	}

	reader, err := gguf.Open(os.Args[1])
	if err != nil {
		log.Fatalf("open gguf: %v", err)
	}
	defer reader.Close()

	rv := reflect.ValueOf(reader).Elem().FieldByName("metadata")
	metadata := reflect.NewAt(rv.Type(), unsafe.Pointer(rv.UnsafeAddr())).Elem()
	iter := metadata.MapRange()
	for iter.Next() {
		key := iter.Key().String()
		val := iter.Value().Interface().(gguf.Metadata).Value
		fmt.Printf("%s: %v\n", key, val)
	}
}
