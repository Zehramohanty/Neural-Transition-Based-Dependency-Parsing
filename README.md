# Neural-Transition-Based-Dependency-Parsing
Built a neural dependency parser using PyTorch. Implemented and trained the
dependency parser, a neural-network based dependency parser, with the goal of maximizing
performance on the UAS (Unlabeled Attachment Score) metric.
This required PyTorch without CUDA installed. 
A dependency parser analyzes the grammatical structure of a sentence, establishing relationships between head
words, and words which modify those heads. This implementation will be a transition-based parser, which incrementally
builds up a parse one step at a time. At every step it maintains a partial parse, which is represented as
follows:

• A stack of words that are currently being processed.

• A buffer of words yet to be processed.

• A list of dependencies predicted by the parser.

Initially, the stack only contains ROOT, the dependencies list is empty, and the buffer contains all words of the
sentence in order. At each step, the parser applies a transition to the partial parse until its buffer is empty and the
stack size is 1. The following transitions can be applied:

• SHIFT: removes the first word from the buffer and pushes it onto the stack.

• LEFT-ARC: marks the second (second most recently added) item on the stack as a dependent of the first item
and removes the second item from the stack.

• RIGHT-ARC: marks the first (most recently added) item on the stack as a dependent of the second item and
removes the first item from the stack.

On each step, the parser will decide among the three transitions using a neural network classifier.
