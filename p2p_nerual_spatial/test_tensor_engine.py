"""Small self-tests for tensor_engine.py; run with python test_tensor_engine.py."""
import numpy as np

from tensor_engine import TensorEngine, TensorEngineError, multiply, add


def main():
    with TensorEngine(device="cpu") as engine:
        a = engine.create_tensor()
        b = engine.create_tensor()
        engine.store(a, (2, 2), [[1, 2], [3, 4]])
        engine.store(b, (2, 2), [[10, 20], [30, 40]])
        c = engine.queue_op(multiply, a, b)
        d = engine.queue_op(add, c, a)
        result = engine.load(d, (2, 2))
        np.testing.assert_allclose(result, [[11, 42], [93, 164]])

        e = engine.queue_op("sum", d, axis=0)
        np.testing.assert_allclose(engine.load(e), [104, 206])

        f = engine.queue_op("reshape", d, shape=(4,))
        np.testing.assert_allclose(engine.load(f, (4,)), [11, 42, 93, 164])

        g = engine.create_tensor((2, 2), fill=3.0)
        np.testing.assert_allclose(engine.load(g), np.full((2, 2), 3.0))

        # Fire-and-forget errors are reported at the next load fence.
        bad = engine.queue_op("definitely_not_an_op", a)
        try:
            engine.load(bad)
        except TensorEngineError:
            pass
        else:
            raise AssertionError("expected an asynchronous operation error at load fence")

    print("TensorEngine CPU self-test: PASS")


if __name__ == "__main__":
    main()
