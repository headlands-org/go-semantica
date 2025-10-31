package annoy

func cloneTrees(src []*node) []*node {
	out := make([]*node, len(src))
	for i, n := range src {
		out[i] = cloneNode(n)
	}
	return out
}

func cloneNode(n *node) *node {
	if n == nil {
		return nil
	}
	cl := &node{
		leaf:       n.leaf,
		hyperplane: append([]float32(nil), n.hyperplane...),
		threshold:  n.threshold,
		indices:    append([]int(nil), n.indices...),
	}
	cl.left = cloneNode(n.left)
	cl.right = cloneNode(n.right)
	return cl
}
