import argparse

from lib.facts.duplicator import DatasetDuplicator
from lib.facts.parser import DatasetParser
from lib.facts.printer import DatasetPrinter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_examples")
    parser.add_argument("out_examples")
    parser.add_argument("--times", "-t", required=True, type=int)
    parser.add_argument("--queries", "-q", nargs=2, type=str, default=None)
    args = parser.parse_args()

    with open(args.in_examples, "r") as inp, open(args.out_examples, "w") as outp:
        parser = DatasetParser()
        dataset_iter = parser.parse_file(inp)
        duplicator = DatasetDuplicator(term_prefix="newterm")
        new_dataset_iter = duplicator.duplicate_samples(dataset_iter, times=args.times, include_orig=True)
        printer = DatasetPrinter(outp)
        printer.print_dataset(new_dataset_iter)

    if args.queries is not None:
        in_queries, out_queries = args.queries

        with open(in_queries, 'r') as inp, open(out_queries, 'w') as outp:
            for _ in range(args.times):
                inp.seek(0)
                for line in inp:
                    outp.write(line)
