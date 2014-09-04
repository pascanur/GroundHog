import cPickle as pkl
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input",
                        type=argparse.FileType('rb'),
                        help="The dictionary to be inverted")
    parser.add_argument("output",
                        type=argparse.FileType('w'),
                        help="The inverted dictionary")
    args = parser.parse_args()

    with open(args.input.name, args.input.mode) as f:
        vocab = pkl.load(f)
        ivocab = dict()
        for k, idx in vocab.iteritems():
            ivocab[idx] = k
        with open(args.output.name, args.output.mode) as fout:
            pkl.dump(ivocab, fout)

if __name__ == '__main__':
    main()
