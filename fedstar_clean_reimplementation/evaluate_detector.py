"""Final evaluation is separate from optimization and requires explicit opt-in."""
from train_detector import main

if __name__ == '__main__':
    main(final=True)
