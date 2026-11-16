import json
import zipfile
from trainingprogram.common import write_json
from trainingprogram.reporting import export_bundle


def test_export_includes_only_named_model_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv('TP_DATA',str(tmp_path))
    model=tmp_path/'models/2B'
    write_json(model/'BF16.json',{'sha256':'weights-hash','command':['convert','--outtype','bf16']})
    write_json(model/'source-manifest.json',{'revision':'fixed','api_key':'do-not-export'})
    write_json(model/'private.json',{'password':'irrelevant'})
    (model/'BF16.gguf').write_bytes(b'weights-not-results')
    target=tmp_path/'results.zip'
    export_bundle(target)
    with zipfile.ZipFile(target) as bundle:
        assert 'models/2B/BF16.json' in bundle.namelist()
        assert 'models/2B/private.json' not in bundle.namelist()
        assert 'models/2B/BF16.gguf' not in bundle.namelist()
        assert json.loads(bundle.read('models/2B/source-manifest.json'))['api_key']=='[REDACTED]'
