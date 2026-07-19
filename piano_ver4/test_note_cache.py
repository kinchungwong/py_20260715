import pytest
import numpy as np
from piano_data_model import PianoCfg, PianoKey, NoteState, LinearRamp, PianoNote
from piano_model import PianoModel, NotePartials
from note_cache import NoteCache, RowBuffer
from time import perf_counter_ns
import cProfile

def test_note_cache_ctor_smoke():
    # This test is a smoke test to ensure that the NoteCache constructor does not raise any exceptions.
    # It does not verify the correctness of the NoteCache's behavior.
    from piano_model import PianoModel, PianoKey
    test_nsamps = 1024
    cfg = PianoCfg()
    model = PianoModel(cfg=cfg)
    for note_id, key in model.keys.items():
        try:
            _ = NoteCache(model, key, test_nsamps)
        except Exception as e:
            pytest.fail(f"NoteCache constructor raised an exception for note_id {note_id}: {e}")

def test_note_cache_render_to():
    # This test is a smoke test to ensure that the NoteCache constructor does not raise any exceptions.
    # It does not verify the correctness of the NoteCache's behavior.
    print("\n\n\n\n")
    from piano_model import PianoModel, PianoKey
    test_nsamps = 1024
    cfg = PianoCfg()
    model = PianoModel(cfg=cfg)
    cfg_lr = LinearRamp(cfg.sample_rate, cfg.attack_msecs, cfg.release_msecs)
    row_buffer = RowBuffer(entries=10, nsamps=test_nsamps, dtype=np.float32)
    out = np.zeros(test_nsamps, dtype=np.float32)
    for note_id, key in model.keys.items():
        try:
            note_cache = NoteCache(model, key, test_nsamps, row_cache=row_buffer)
            note_state = NoteState(cfg, cfg_lr, NotePartials(model, key))
            note = PianoNote(note_id=note_id, velocity=127)
            note_state.attack(note)
            t0 = perf_counter_ns()
            note_cache.render_to(note_state=note_state, out=out)
            t1 = perf_counter_ns()
            value_0 = out[0]
            maxval = np.max(out)
            minval = np.min(out)
            print(f"NoteCache.render_to for note_id {note_id}: maxval={maxval:.2f}, minval={minval:.2f}, value[0]={value_0:.2f}, elapsed={(t1 - t0)*1e-6:.6f} ms")
        except Exception as e:
            pytest.fail(f"NoteCache.render_to raised an exception for note_id {note_id}: {e}")

if __name__ == "__main__":
    exit(pytest.main([__file__, "-v", "-s"]))
    # cProfile.run("test_note_cache_render_to()")
