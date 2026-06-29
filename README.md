# micrograd from scratch :

So this is my implementation of Andrej Karpathy's micrograd — basically I built a tiny autograd engine + neural net library from scratch, no PyTorch, no TensorFlow, just plain Python.

The whole point of this project wasn't to build something fancy or production-ready. It was to actually understand what's happening when you call `.backward()` in PyTorch instead of just trusting it blindly. Turns out it's not magic at all — it's just the chain rule applied really carefully, over and over, across a graph.

## what's actually in here :

I built a `Value` class that wraps regular numbers and tracks how they were created — so when you do `a * b`, it doesn't just give you the answer, it remembers that this number came from a multiplication of `a` and `b`. Do enough of these operations and you end up with a full computation graph, built automatically, just from doing normal math.

Once that graph exists, I implemented `.backward()` — which walks the graph in reverse and computes gradients for every single node using the chain rule. This is backpropagation, the algorithm that literally every neural network (yes including GPT) uses to learn.

On top of the `Value` engine, I built an actual neural net library — `Neuron`, `Layer`, `MLP` classes, basically copying the same API style PyTorch uses (`__call__` for forward pass, `parameters()` to grab all the weights). And then trained a tiny network on a toy dataset and watched the loss actually go down. That part felt really satisfying after debugging gradients for hours.

I also cross-checked everything against real PyTorch — same inputs, same operations — and got identical gradients. That was honestly the most reassuring part, knowing the math wasn't just working by coincidence.

## why I did this :

I'm currently going through a structured AI/ML learning track (deep learning phase specifically), and this video is basically the gold standard starting point everyone recommends before touching real frameworks. I didn't want to just watch and nod along — I wanted to type out every line, break it, fix it, and actually internalize what backprop is doing under the hood.

## a couple of notes :

- No graphviz visualization here — built this on a locked-down corporate laptop with no admin rights, so the system Graphviz binary wasn't installable. Wrote a simple text-based graph printer instead that does basically the same job.
- Everything's pip-installable, nothing fancy needed beyond `numpy`, `matplotlib`, and `torch` (only used for the comparison check, not required for the engine itself).

## what's next :

This is project 1 of a few deep learning builds I'm working through — CNNs, RNN→LSTM→Transformer progression, and self-supervised learning are next on the list. Micrograd was step zero, basically making sure the foundation is actually solid before stacking more complexity on top of it.

## credit : 

All credit for the original idea and teaching goes to Andrej Karpathy — [the video](https://www.youtube.com/watch?v=VMj-3S1tku0) is genuinely one of the best explanations of backprop out there. This repo is my own typed-out, debugged, and slightly modified version of following along with it.
