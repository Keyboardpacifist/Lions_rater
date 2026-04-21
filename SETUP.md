# Setting Up Local Development

This guide gets you from zero to running the app locally in one session. Every command is copy-pasteable. If something fails, paste the error to Claude and it will help you fix it.

## 1. Open Terminal

Press **Cmd + Space**, type **Terminal**, press Enter.

You'll see something like:

```
brett@macbook ~ %
```

That's your command prompt. Everything below that starts with `$` means "type this in Terminal." Don't type the `$` itself.

## 2. Install Homebrew (Mac package manager)

Check if you already have it:

```bash
$ brew --version
```

If you see a version number, skip ahead. If you see "command not found," install it:

```bash
$ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow the prompts. When it finishes, it may tell you to run two commands to add brew to your PATH — **do what it says.**

## 3. Install Python and Git

```bash
$ brew install python git
```

Verify:

```bash
$ python3 --version
$ git --version
```

Both should print version numbers.

## 4. Set up Git identity

Tell Git who you are (use the same email as your GitHub account):

```bash
$ git config --global user.name "Your Name"
$ git config --global user.email "your-email@example.com"
```

## 5. Set up SSH key for GitHub

This lets you push/pull without typing your password every time.

```bash
$ ssh-keygen -t ed25519 -C "your-email@example.com"
```

Press Enter three times to accept defaults (no passphrase is fine for personal use).

Now copy the public key:

```bash
$ cat ~/.ssh/id_ed25519.pub | pbcopy
```

That copied it to your clipboard. Now:

1. Go to https://github.com/settings/keys
2. Click **New SSH key**
3. Title: "My Mac"
4. Paste the key (Cmd + V)
5. Click **Add SSH key**

Test it:

```bash
$ ssh -T git@github.com
```

You should see: "Hi Keyboardpacifist! You've successfully authenticated..."

## 6. Clone the repo

Pick a folder for your projects. The home directory is fine:

```bash
$ mkdir -p ~/Sites
$ cd ~/Sites
$ git clone git@github.com:Keyboardpacifist/Lions_rater.git
$ cd Lions_rater
```

You now have the entire repo on your machine.

## 7. Install dependencies and run

```bash
$ make install
```

This creates a virtual environment and installs everything. Takes about a minute.

Now create your secrets file so Supabase works locally:

```bash
$ mkdir -p .streamlit
```

Then create the file `.streamlit/secrets.toml` with your Supabase credentials. You can do this in any text editor, or ask Claude to help you create it. It should contain:

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-anon-key"
```

(These are the same values that are in your Streamlit Cloud dashboard under Settings > Secrets.)

Now run the app:

```bash
$ make run
```

Open http://localhost:8501 in your browser. You should see your app running locally.

**Leave this terminal tab running.** Every time you save a file, Streamlit auto-reloads. You'll see changes instantly.

## 8. Run the tests

Open a **new** terminal tab (Cmd + T):

```bash
$ cd ~/Sites/Lions_rater
$ make test
```

You should see 85 tests pass. This validates your scoring math and data files.

## 9. The new workflow

**Before (the copy-paste way):**
1. Edit code somewhere
2. Copy to GitHub web UI
3. Commit via browser
4. Wait for Streamlit Cloud to redeploy
5. Check if it works
6. If not, repeat from 1

**After (the local way):**
1. Edit code in your editor
2. Save the file
3. Browser auto-refreshes — see the result in 2 seconds
4. When it works, commit and push:

```bash
$ git add the-file-you-changed.py
$ git commit -m "Describe what you changed"
$ git push
```

Streamlit Cloud auto-deploys from your push. Done.

## 10. Useful commands

| Command | What it does |
|---------|-------------|
| `make run` | Run the app locally |
| `make test` | Run the test suite |
| `make lint` | Check code style |
| `git status` | See what files you've changed |
| `git diff` | See exactly what changed |
| `git add -A` | Stage all changes |
| `git commit -m "message"` | Save a checkpoint |
| `git push` | Push to GitHub (triggers deploy) |
| `git pull` | Get latest from GitHub |

## Troubleshooting

**"Permission denied" on git push:**
Your SSH key isn't set up correctly. Go back to step 5.

**"Module not found" error when running:**
Run `make install` again.

**App runs but shows "missing secrets" error:**
Your `.streamlit/secrets.toml` is missing or has wrong values. See step 7.

**"Port 8501 already in use":**
You have another Streamlit running. Find it with `lsof -i :8501` and kill it, or use `make run` which handles this.

**Something else broke and you're stuck:**
```bash
$ git stash        # Saves your changes temporarily
$ git pull         # Gets the latest working version
$ make run         # Try running again
$ git stash pop    # Brings your changes back
```

Or just paste the error into Claude. That's what it's here for.
