#!/usr/bin/env bash
set -e

# Install all deps from requirements.txt using prebuilt wheels only
# (HPC has GCC 7.3 and no Rust compiler, so source builds fail)
uv pip install -r requirements.txt --extra-index-url https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Patch orbit_wars env files into kaggle-environments 1.12.0
# (orbit_wars was added in 1.28.0 which can't install on this HPC)
ENVS_DIR=".venv/lib/python3.11/site-packages/kaggle_environments/envs"
WHEEL_URL="https://files.pythonhosted.org/packages/$(curl -s 'https://pypi.org/pypi/kaggle-environments/1.28.0/json' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); w=[f for f in d['urls'] if f['filename'].endswith('.whl')]; print(w[0]['url'].split('/packages/')[1])")"

mkdir -p "$ENVS_DIR/orbit_wars"
curl -sL "https://files.pythonhosted.org/packages/$WHEEL_URL" -o /tmp/kenv_1.28.whl
python3 -c "
import zipfile
z = zipfile.ZipFile('/tmp/kenv_1.28.whl')
for name in z.namelist():
    if name.startswith('kaggle_environments/envs/orbit_wars/'):
        fname = name.split('/')[-1]
        with open('$ENVS_DIR/orbit_wars/' + fname, 'wb') as f:
            f.write(z.read(name))
        print('Extracted:', fname)
"
rm /tmp/kenv_1.28.whl

echo "Done. Verify with: .venv/bin/python -c \"from kaggle_environments import make; make('orbit_wars'); print('OK')\""
