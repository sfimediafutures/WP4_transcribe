import subprocess
import fcntl
import os
import select


ccmodule = {
    "description": "Transcribe a media file",
    "depends": [],
    "provides": [],
    "inputs": {
        "src": "Source file to transcribe",
        "model": "tiny.en, tiny, base.en, base, small.en, small medium.en, medium, large. Default large",
        "task": "transcribe or translate (to english)",
        "lang": "Language, default autodetect",
        "dir": "Directory to place files"
    },
    "outputs": {
        "dst": "Output file (VTT)",
        "dst_txt": "Output file (Text)"

    },
    "defaults": {
        "priority": 50,  # Normal
        "runOn": "success"
    },
    "status": {
        "progress": "Progress 0-100%",
        "state": "Current state of processing"
    }
}


def process_task(cc, task, stop_event):

    print("MOD_WHISPER", task)

    args = task["args"]

    model = args.get("model", "large")
    lang = args.get("lang", None)
    src = args["src"]
    dst_dir = args.get("dir", "/tmp")

    cmd = ["whisper", "--model", model, "--patience", "0.01", "-o", dst_dir, "--task"]

    cmd.append(args.get("task", "transcribe"))
    if lang:
        cmd.extend(["--language", lang])

    cmd.append(src)
    base_dst = os.path.join(dst_dir, os.path.basename(src))
    retval = {
        "dst": base_dst + ".vtt",
        "dst_txt": base_dst + ".txt"
    }

    dst = retval["dst"]
    # As long as caching doesn't work, check for existence of file
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        cc.log.warning("Cache failed to catch this one")
        return 100, retval

    print("Will run")
    print(" ".join(cmd))

    cc.log.debug(" ".join(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    fcntl.fcntl(p.stdout, fcntl.F_SETFL, os.O_NONBLOCK)
    fcntl.fcntl(p.stderr, fcntl.F_SETFL, os.O_NONBLOCK)
    progress = 0
    while not stop_event.isSet():

        if p.poll() is not None:
            # Done
            if p.poll() == 0:
                progress = 100
                retval["result"] = "ok"
                break
            raise Exception("Whisper failed with exit value %s" % p.poll())

        ready = select.select([p.stdout, p.stderr], [], [], 1.0)[0]
        for fd in ready:

            if fd == p.stdout:
                # Rather do some progress here?
                msg = fd.read().strip()
                if msg:
                    cc.log.debug(msg)
            else:
                msg = fd.read().strip()
                if msg:
                    cc.log.warning(msg)

    if stop_event.isSet():
        try:
            cc.log.info("Terminating Whisper")
            p.terminate()
        except Exception:
            pass

    return progress, retval
