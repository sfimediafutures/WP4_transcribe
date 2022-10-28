import json
import operator
import time
import re
import math
import copy
import os
import pickle
import base64

ccmodule = {
    "description": "Identify speakers, requires 16khz audio",
    "depends": [],
    "provides": [],
    "inputs": {
        "src": "Media source file",
        "vtt": "Subtitle file (created by Whisper for example)",
        "segments": "Detected audio segments",
        "people": "Already known list of people (if available) [people_dir/name or path, ...]",
        "people_dir": "Directory of people (if not absolute paths in people file)",
        "guess_people": "If people are given, still guess for others? Default True",
        "dst": "Destination subtitle json file",
        "cutoff": "How close match to regard as a person - default 0.1, higher number = more closely"
    },
    "outputs": {
        "cast": "JSON file with cast members",
        "dst": "JSON file with updated segments"
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

try:
    import nemo.collections.asr as nemo_asr
    import torch
    import whisper
    CANRUN = True
except Exception:
    CANRUN = False


class WhisperBits():
    @staticmethod
    def save_segment(source, start, end, max_length=30, trim_percent=0):
        """
        Create a temporary file  with the given segment
        max_len will limit the max size

        trim_percent will take off the percent (in total) divided by half at
        the beginning and half at the end.
        """
        end = start + min(max_length, end - start)
        length = end - start

        if trim_percent and length > 1.0:  # We only trim if over a second
            start += length * ((trim_percent / 2) / 100.)
            end -= length * ((trim_percent / 2) / 100.)

        import wave
        import tempfile
        dst_file = tempfile.mktemp(suffix=".wav")
        dst = wave.open(dst_file, "w")
        src = wave.open(source, "r")

        # Skip to position
        rate = src.getframerate()
        # Sanity
        if start > src.getnframes() / rate:
            raise Exception("Segment starts after file end, %s, %s" % (start, source))

        src.setpos(math.floor(rate * start))
        data = src.readframes(math.ceil(rate * (end - start)))
        dst.setsampwidth(src.getsampwidth())
        dst.setnchannels(src.getnchannels())
        dst.setframerate(rate)
        dst.writeframes(data)
        return dst_file

    def __init__(self, sourcefile, model="large"):

        self.cache = {}
        self.cachefile = "whisper_cache.json"
        if os.path.exists(self.cachefile):
            with open(self.cachefile, "r") as f:
                self.cache = json.load(f)
        self.model = whisper.load_model(model)

        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(sourcefile)
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

        # detect the spoken language
        _, probs = self.model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")

        self.options = whisper.DecodingOptions()

    def __del__(self):
        with open(self.cachefile, "w") as f:
            self.cache = json.dump(self.cache, f)

    def decode(self, sourcefile, start, end):
        """
        Analyse a given bit of a file
        """

        if str((sourcefile, start, end)) in self.cache:
            return self.cache[str((sourcefile, start, end))]

        f = self.save_segment(sourcefile, start, end, max_length=100)
        audio = whisper.load_audio(f)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        result = whisper.decode(self.model, mel, self.options)
        self.cache[str((sourcefile, start, end))] = result.text
        os.remove(f)
        return result.text


class VoiceCompare():
    def __init__(self, log):

        self.model = None
        self.last_model_id = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.log = log
        self.cast = []  # We group "seconds" for each speaker
        self.detected_people = {}
        self._cast = {}

    def _load_model(self, model_id="nvidia/speakerverification_en_titanet_large"):

        if self.last_model_id != model_id:
            # Free existing?
            self.last_model_id = model_id
            self.model = speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_id)

    def compare_embeddings(self, embeddings0, embeddings1):
        # the resulting embeddings can be used for cosine similarity-based retrieval
        cosine_sim = torch.nn.CosineSimilarity(dim=-1)
        similarity = cosine_sim(embeddings0, embeddings1)
        return similarity

    def get_embedding(self, wavfile, start=None, end=None):
        """
        If start and end are given, a temporary file is created and embeddings returned
        """
        self._load_model()

        if start is not None and end is not None:
            try:
                f = self.save_segment(wavfile, start, end, max_length=10000)
                return self.model.get_embedding(f).to(self.device)
            finally:
                try:
                    os.remove(f)
                except Exception:
                    pass
        return self.model.get_embedding(wavfile).to(self.device)

    @staticmethod
    def save_segment(source, start, end, max_length=1, trim_percent=0):
        """
        Create a temporary file  with the given segment
        max_len will limit the max size

        trim_percent will take off the percent (in total) divided by half at
        the beginning and half at the end.
        """
        end = start + min(max_length, end - start)
        length = end - start

        if trim_percent and length > 1.0:  # We only trim if over a second
            start += length * ((trim_percent / 2) / 100.)
            end -= length * ((trim_percent / 2) / 100.)

        import wave
        import tempfile
        dst_file = tempfile.mktemp(suffix=".wav")
        dst = wave.open(dst_file, "w")
        src = wave.open(source, "r")

        # Skip to position
        rate = src.getframerate()
        # Sanity
        if start > src.getnframes() / rate:
            raise Exception("Segment starts after file end, %s, %s" % (start, source))

        src.setpos(math.floor(rate * start))
        data = src.readframes(math.ceil(rate * (end - start)))
        dst.setsampwidth(src.getsampwidth())
        dst.setnchannels(src.getnchannels())
        dst.setframerate(rate)
        dst.writeframes(data)
        return dst_file

    @staticmethod
    def read_csv(filename):
        entries = []
        with open(filename, "r") as f:
            for line in f.readlines():
                if line.count(",") == 5:
                    speaker, start, end, duration, fn, text = line.split(",", 5)
                else:
                    speaker, start, end, duration, fn = line.split(",", 4)
                    text = ""

                if not start.replace(".", "").isnumeric():
                    continue  # Bad line

                sub = {
                    "start": float(start),
                    "end": float(end),
                    "who": speaker,
                    "file": fn,
                    "text": text
                }
                entries.append(sub)
        return entries

    def load_embeddings(self, wavfile, segments, min_time=None, max_time=None):

        embeddings = []
        for segment in segments:
            if min_time and segment["start"] < min_time:
                continue
            if max_time and segment["end"] > max_time:
                break

            # if segment["end"] > 900:
            #   break  # For testing
            for i in range(0, 2 * math.ceil(segment["end"] - segment["start"])):
                e = min(segment["start"] + 0.5 + (0.5 * i), segment["end"])
                if e - (segment["start"] + (0.5 * i)) < 0.1:
                    continue  # Too short
                try:
                    embedding = self.get_embedding(wavfile, segment["start"] + (i * 0.5), e)
                    embeddings.append(((segment["start"] + (0.5 * i), e), embedding))
                except Exception as ex:
                    if self.log:
                        self.log.exception("Failed to get embeddings for %s [%s - %s]" %
                                           (wavfile, segment["start"] + (0.5 * i), e))
                    else:  # Only while debugging
                        print("Failed to get embeddings for %s [%s - %s]" %
                              (wavfile, segment["start"] + (0.5 * i), e))
                        import traceback
                        traceback.print_exc()
        return embeddings

    def find_best_matches(self, embeddings, safe_hit=0.35):
        cast = []
        best_matches = []
        for x, e0 in enumerate(embeddings):

            # If it's a good match with the previous one, go for that
            if 0 and x > 0:
                s = vc.compare_embeddings(e0[1], embeddings[x - 1][1])
                if s >= safe_hit:
                    print("Quick-hit", x, s)
                    best_matches.append([x, x - 1, s])
                    continue

            max_sim = [None, 0]
            for y, e1 in enumerate(embeddings):
                if y < x + 1:
                    continue
                sim = vc.compare_embeddings(e0[1], e1[1])
                if sim > max_sim[1]:
                    max_sim = [y, sim]
                if sim >= safe_hit:
                    break

            best_matches.append((x, max_sim[0], max_sim[1]))
        return best_matches

    def get_embeddings_by_time(self, embeddings, starttime, endtime):
        """
        Return a loaded embedding for the given time
        """

        ret = []
        for ts, embedding in embeddings:
            if ts[0] >= starttime and ts[1] <= endtime:
                ret.append((ts, embedding))
        return ret

    def guess_known_items(self, wavfile, segments, _embeddings, starttime=None,
                          endtime=None):
        """
        Go through the given segments, find some long ones and determine
        if they are one person only.
        If so, is this a known person?
        Will use/update the detected_people map
        """
        print("Guessing known items, %d already known" % len(self.detected_people))
        min_segment_length = 5  # seconds
        padding = 0.5  # We skip the first and last bits in case it's no good
        cutoff = 0.20  # CHECK THIS ONE
        self._people_times = {}

        for segment in segments:
            if starttime and segment["start"] < starttime:
                continue
            if endtime and segment["end"] > endtime:
                break

            start = segment["start"] + padding
            end = segment["end"] - padding
            duration = end - start
            if duration < min_segment_length:
                continue

            # Get the embeddings for this time
            embeddings = self.get_embeddings_by_time(_embeddings, start, end)
            # embeddings = self.load_embeddings(wavfile, [{"start": start, "end": end}])

            # Are all the embeddings the same person?
            matches = []
            for x in range(0, len(embeddings)):
                for y in range(x + 1, len(embeddings)):
                    matches.append(self.compare_embeddings(embeddings[x][1], embeddings[y][1]))
            avg = sum(matches) / max(1, len(matches))
            # print("AVG", avg)
            if avg > cutoff:  # GUESSING that this is ok
                # Is this a known person?
                found = False
                for p in self.detected_people:
                    for e in self.detected_people[p]:
                        m = []
                        for x in embeddings:
                            m.append(self.compare_embeddings(e[1], x[1]))
                        print("AVG ", p, sum(m) / max(1, len(m)), "cutoff", cutoff)
                        if sum(m) / max(1, len(m)) > cutoff:
                            found = True
                            print("FOUND PERSON", p, len(self.detected_people[p]))
                            if len(self.detected_people[p]) > 20:
                                break
                            self.detected_people[p].extend(embeddings)
                            print("Now have %d people" % len(self.detected_people))
                            break

                if not found:
                    print("New person")
                    self.detected_people[len(self.detected_people)] = embeddings

            if len(matches) > 0:
                print("SEGMENT [%s-%s] %.2f - %.2f" % (start, end, min(matches), sum(matches) / len(matches)))

        return self.detected_people

    def cast_to_people(self, castfile):
        """
        Loads embeddings if any and store them into self.detected_people
        """

        with open(castfile, "r") as f:
            cast = json.load(f)

        # If any segments are defined, load them
        people = {}
        for member in cast:
            if "segments" in cast[member]:
                people[member] = []
                for s in cast[member]["segments"]:
                    sourcefile = s["file"]
                    times = s["times"]
                    for start, end in times:
                        delta = 1.0
                        for i in range(0, 2 * math.ceil(end - start)):
                            s = start + (i * delta)
                            e = min(s + delta, end)
                            if e - (start + (delta * i)) < 0.1:
                                continue  # Too short
                        embedding = self.get_embedding(wavfile, s, e)
                        people[member].append(((s, e), embedding))

        self.detected_people = people
        self._cast = cast  # Save if we convert back and have more info
        return people

    def people_to_cast(self, wavfile, people=None):
        """
        Use to add auto-detected people
        """
        if not people:
            people = self.detected_people

        cast = self._cast

        for person in people:
            # This is a list of ((start, stop), embeddings) for .5 or 1s clips
            # Gather them and bundle into segments

            if person in self._cast:
                print("Already in cast", person)
                continue  # Already present

            segments = []
            segs = [ts for ts, _ in people[person]]
            for idx, s in enumerate(segs):
                if idx == 0:
                    segments = [[s[0], s[1]]]
                    continue
                if segments[-1][1] == s[0]:  # If the last end is equal to this start join
                    segments[-1][1] = s[1]
                else:
                    segments.append([s[0], s[1]])
            self._cast[person] = {
                "name": person,
                "color": "#ffeeffA6",  # TODO: Randomize
                "segments": [{"file": wavfile, "times": segments}]
            }

        return self._cast

    def load_person_list(self, people_json_file, person_dir=""):
        """
        person_json_file is a list [person1, person2, person3]
        # where person is a name to be found in the people_dir or 

        The person info file is json:    
        Filename is json with 'name' and 'voice', where 'voice' is base64
        encoded pickled embeddings.
        Returns map of embeddings for known people, stores people into self._cast too
        """

        with open(people_json_file, "r") as f:
            ppl = json.load(f)

        known_items = {}

        for p in ppl:
            filename = p
            if not os.path.exists(p):
                if not person_dir or not os.path.exists(os.path.join(person_dir, p)):
                    raise Exception("Missing person '%s'" % p)
                else:
                    filename = os.path.join(person_dir, p)
            else:
                filename = p

            with open(filename, "r") as f:
                info = json.load(f)
            embeddings = pickle.loads(base64.b64decode(info["voice"].encode("ascii")))
            del info["voice"]

            known_items[info["name"]] = [x.to(self.device) for x in embeddings]
            self._cast[info["name"]] = info
            self.detected_people[info["name"]] = [[[0, 1], x.to(self.device)] for x in embeddings]

        return known_items

    def build_known_items(self, known_people):
        """
        We have a map of people and a list of times they are represented in
        {'person1':  {"segments": [{"file": ..., "times": [(0.1, 5.0), (10.1, 15.4)]}}]}}
        """
        p = {}
        for person in known_people:
            p[person] = []
            if "segments" not in known_people[person]:
                continue
            for segment in known_people[person]["segments"]:
                # for segment in segments:
                if 1:
                    wavfile = segment["file"]
                    for start, end in segment["times"]:
                        if 0:
                            # Split into 1s clips
                            delta = 1.0
                            for i in range(0, 2 * math.ceil(end - start)):
                                e = min(start + delta + (delta * i), end)
                                if e - (start + (delta * i)) < 0.1:
                                    continue  # Too short
                                embedding = self.get_embedding(wavfile, start + (i * delta), e)
                                p[person].append(embedding)

                        else:  # We use the whole segment
                            embedding = self.get_embedding(wavfile, start, end)
                            p[person].append(embedding)

                # As an alternative, create a list of 0.5s bits?
        return p

    def find_best_matches_known_items(self, known_items, embeddings, safe_hit=0.35):
        cast = []
        best_matches = []
        for x, e0 in enumerate(embeddings):

            max_sim = [None, 0]
            total = 0
            for person in known_items:
                for y, e1 in enumerate(known_items[person]):
                    sim = self.compare_embeddings(e0[1], e1)
                    if sim > max_sim[1]:
                        max_sim = [person, sim]
                    # total += sim
                    if sim >= safe_hit:
                        break
                # avg = total / len(known_items[person])
                # if avg > max_sim[1]:
                #    max_sim = [person, avg]

            best_matches.append((x, max_sim[0], max_sim[1]))
        return best_matches

    def build_cast_known_items(self, embeddings, known_items, best_matches, cutoff):
        # Collect into cast members
        cast = {}
        for x, person, sim in best_matches:
            if sim < cutoff:
                continue

            if person not in cast:
                cast[person] = []

            cast[person].append(x)

        # Merge blocks
        def merge_blocks(lst):
            ret = []
            # lst2 = [embeddings[i][0] for i in lst]
            # lst2.sort(key=operator.itemgetter(0))
            for i, item in enumerate(lst):
                start, end = embeddings[item][0]
                if len(ret) > 0 and abs(start - ret[-1]["end"]) <= 0.1:
                    ret[-1]["end"] = end
                else:
                    ret.append({"start": start, "end": end})
            return ret

        # Cleanup

        # For each cast member, there is a list of time slots, we merge the
        # slots into blocks
        for person in cast:
            # print("PERSON", person, cast[person])
            cast[person] = merge_blocks(cast[person])

        # cast = [sorted(merge_blocks(cast[m]), key=operator.itemgetter("start")) for m in cast if len(m) > 0]

        return cast

    def build_cast(self, best_matches, cutoff):
        # Collect into cast members
        cast = []
        for x, y, sim in best_matches:
            if sim < cutoff:
                continue

            found = False
            for member in cast:
                if y in member and x not in member:
                    member.append(x)
                    found = True
                if x in member and y not in member:
                    member.append(y)
                    found = True

            if not found:
                # No hit, new cast
                cast.append([x, y])

        def unique(lst):
            """
            Returns all the unique items of a list
            """
            u = []
            for i in lst:
                if i not in u:
                    u.append(i)
            return u

        # We might have various "blocks" of similarity now, merge them
        while True:
            any_merged = False
            for x, outer in enumerate(cast):
                merged = False
                for y, inner in enumerate(cast[x + 1:]):
                    y += x + 1
                    for item in outer:
                        if item in inner:
                            # The inner item overlaps with the outer - merge inner into outer and zero it
                            outer.extend(inner)
                            cast[x] = unique(outer)
                            cast[y] = []
                            merged = True
                            break
                    if merged:
                        any_merged = True
                        break
            if not any_merged:
                break

        # Merge blocks
        def merge_blocks(lst):
            ret = []
            lst2 = [embeddings[i][0] for i in lst]
            lst2.sort(key=operator.itemgetter(0))
            for i, item in enumerate(lst2):
                start, end = item  # embeddings[item][0]
                if len(ret) > 0 and abs(start - ret[-1]["end"]) <= 0.1:
                    ret[-1]["end"] = end
                else:
                    ret.append({"start": start, "end": end})
            return ret

        # Cleanup
        cast = [sorted(merge_blocks(m), key=operator.itemgetter("start")) for m in cast if len(m) > 0]

        return cast

    def find_most_likely_speaker(self, cast, start, end):
        """
        Look through the cast list, find all the members that are active
        within start-end and return the most likely cast member (if any).
        """
        active_members = {}
        for person in cast:
            active_members[person] = 0
            for entry in cast[person]: # entry has "start", "end"
                if entry["end"] > start and entry["start"] < end:
                    # Add the overlapping part of this segment
                    active_members[person] += min(entry["end"], end) - max(entry["start"], start)
                if entry["end"] > end:
                    break  # No more hits for this member

        l = [(i, active_members[i]) for i in active_members]
        l.sort(key=operator.itemgetter(1), reverse=True)
        if l[0][1] == 0:
            return None
        return l[0][0]

    def find_most_likely_speaker_OLD(self, cast, start, end):
        """
        Look through the cast list, find all the members that are active
        within start-end and return the most likely cast member (if any).
        """
        active_members = {}
        for mid, member in enumerate(cast):
            active_members[mid] = 0
            for entry in member: # entry has "start", "end"
                if entry["end"] > start and entry["start"] < end:
                    # Add the overlapping part of this segment
                    active_members[mid] += min(entry["end"], end) - max(entry["start"], start)
                if entry["end"] > end:
                    break  # No more hits for this member

        l = [(i, active_members[i]) for i in active_members]
        l.sort(key=operator.itemgetter(1), reverse=True)
        if l[0][1] == 0:
            return None
        return l[0][0]

    def get_match_length(self, str1, str2):
        """
        How much of these match each other (lower case, ignore punctuation etc)
        """

        _str1 = "".join(e for e in str1 if e.isalnum() or e == " ").strip().lower()
        _str2 = "".join(e for e in str2 if e.isalnum() or e == " ").strip().lower()
        i = 0
        for i in range(min(len(_str1), len(_str2))):
            if _str1[i] != _str2[i]:
                if i == 0:
                    return i
                else:
                    return i + 1
        if i == 0:
            return 0
        return i + 1

    def realign_subs_whisper(self, wavfile, subs, speakers, cast, min_time=1.3):
        """
        Use Whisper to check all "speakers". For each speaker segment, it
        transcribes text and checks if a subtitle starts with that text. 
        It also fills in "who" from speakers into the subs.
        """
        MAX_CPS = 25.0
        MIN_CPS = 15.0
        whisper = WhisperBits(wavfile)
        # First we go through the segments and see if we can find the subs
        # that match it - we use this to update the start points.
        subidx = 0
        for entry in speakers:
            # TODO: Get the timestamps back - if it's not at the very start, ignore it!
            text = whisper.decode(wavfile, entry["start"], min(entry["start"] + 8, max(entry["start"] + 2, entry["end"])))
            # text = "".join(e for e in text if e.isalnum() or e == " ").strip().lower()
            print("Looking for", entry["start"], entry["end"], entry["who"])
            print("   ", text)
            # Check the close subtitles
            best_hit = [None, 0]
            for idx, s in enumerate(subs):
                if abs(entry["start"] - s["end"]) > 7 and abs(entry["end"] - s["start"]) > 7:
                    continue
                if "matched" in s:
                    continue

                m = self.get_match_length(text, s["text"]) / float(len(s["text"]))
                print("m?", m, text, s["text"])
                # We need to match at least one word
                # if m > min(len(s["text"]), s["text"].find(" ")) and \
                if m > best_hit[1] and m > 0.8:
                    best_hit = [idx, m]
                continue

            if best_hit[1]:
                idx = best_hit[0]
                MINDUR = len(subs[idx]["text"]) / MAX_CPS
                MAXDUR = len(subs[idx]["text"]) / MIN_CPS
                print(" * HIT %.2f %s [%s, %s]: %s" % (best_hit[1], entry["who"], subs[idx]["start"], subs[idx]["end"], subs[idx]["text"][:25]))
                print("    Duration: min", MINDUR, "max", MAXDUR, "actual", subs[idx]["end"] - subs[idx]["start"])
                # subs[idx]["end"] = max(ESTDUR, min(entry["end"], entry["start"] + subs[idx]["end"] - subs[idx]["start"]))
                subs[idx]["end"] = min(entry["start"] + MAXDUR, max(entry["start"] + MINDUR, entry["start"] + subs[idx]["end"] - subs[idx]["start"]))
                subs[idx]["start"] = entry["start"]
                subs[idx]["who"] = entry["who"]
                subs[idx]["matched"] = True
                print("    -> [%s, %s]" % (subs[idx]["start"], subs[idx]["end"]))

            print()

        print(" ----- After first run -----")
        for s in subs:
            print(s)

        # If we have overlapping items, move them around a bit
        for idx, sub in enumerate(subs):
            if idx == 0:
                continue

            if subs[idx - 1]["end"] > subs[idx]["start"]:
                print("Items overlap\n%s\n%s"% (str(subs[idx - 1]), str(subs[idx])))

                if "who" in subs[idx]:
                    print("   - ignoring - this has been adjusted already")
                    continue

                duration = subs[idx]["end"] - subs[idx]["start"]
                subs[idx]["start"] = subs[idx - 1]["end"]
                if idx < len(subs):
                    subs[idx]["end"] = subs[idx]["start"] + duration
                    # subs[idx]["end"] = min(subs[idx + 1]["start"], subs[idx]["start"] + duration)
                    print("    --> ", subs[idx])
                else:
                    subs[idx]["end"] = max(subs["idx"]["end"], subs[idx]["start"] + 1.5)  # TODO: Use CPS or get the segment?
                    print("    ---> ", subs[idx])

                    # See if we can find the segment?
                    for entry in speakers:
                        if entry["start"] >= subs[idx]["start"]:
                            print("        Possible end is segment", entry["end"])
                            break

        # Fill in missing speakers
        print(" ----- After second run -----")
        for s in subs:
            print(s)
        print("Filling in missing speakers")
        for idx, sub in enumerate(subs):
            # if "who" not in sub:
            subs[idx]["who"] = self.find_most_likely_speaker(cast, sub["start"], sub["end"]) 
            print("[%d]:" % idx, subs[idx])

        print(" ----- After third run -----")
        for s in subs:
            print(s)

        # We do a pass now to ensure that a single person doesn't overlap!
        for idx in range(1, len(subs)):
            if subs[idx - 1]["who"] != subs[idx]["who"]:
                continue
            # Same speaker - do they overlap?
            if subs[idx]["start"] <= subs[idx - 1]["end"]:
                subs[idx]["end"] = subs[idx - 1]["end"] + subs[idx]["end"] - subs[idx]["start"]
                subs[idx]["start"] = subs[idx - 1]["end"]

        print(" ----- After fourth run -----")
        for s in subs:
            print(s)

        print("Validating speakers")
        for idx, sub in enumerate(subs):
            # if "who" not in sub:
            subs[idx]["who"] = self.find_most_likely_speaker(cast, sub["start"], sub["end"]) 
            print("[%d]:" % idx, subs[idx])

        return subs

    def realign_subs(self, subs, _segments, cast, max_adjust=0.6, min_time=1.3):
        """
        subs need "start" and "end", will align both to segments and to cast
        timings
        """

        updated = kept = 0    
        subs = sorted(subs, key=operator.itemgetter("start"))

        segments = copy.copy(_segments)
        # We merge segments and cast-times to estimate potential sync points
        for mid, member in enumerate(cast):
            for t in member:
                t["who"] = mid
                segments.append(t)

        segments.sort(key=operator.itemgetter("start"))

        print(len(segments), "segments", len(subs), "subs")

        for sub in subs:
            found = False
            # Find a start in the segments that is very close, and if found, re-align
            for segment in segments:
                if abs(segment["start"] - sub["start"]) < max_adjust:
                    if not found:
                        sub["start"] = segment["start"]
                        if sub["end"] - sub["start"] < min_time:
                            sub["end"] = sub["start"] + min_time
                        found = True
                        updated += 1
                    else:
                        # Found start, find end too
                        if abs(segment["end"] - sub["end"]) < max_adjust:
                            sub["end"] = max(sub["start"] + min_time, segment["end"])
                            break
            if not found:
                kept += 1

        # Do some additional checking - if two subs close very close to each other, bundle them
        threshold = 0.4
        # If some overlap with a tiny bit, shorten down the first
        for idx, sub in enumerate(subs):
            if idx > 0:
                if abs(sub["end"] - subs[idx-1]["end"]) < threshold:
                    # print("Aligning ends", subs[idx-1], sub)
                    subs[idx-1]["end"] = sub["end"]
                if subs[idx-1]["end"] - sub["start"]  < threshold * 2 and subs[idx-1]["end"] - sub["start"] > 0:
                    # print("Overlapping\n", subs[idx-1],"\n", sub)
                    subs[idx-1]["end"] = sub["start"] - 0.001

        # Sanity
        for sub in subs:
            if sub["end"] < sub["start"]:
                raise SystemExit("Super-wrong, end is before start", sub)

        print("Updated", updated, "kept", kept)

        return subs

    def write_csv(self, segments, dst):
        with open(dst, "w") as f:
            f.write("speaker,start,end,duration,audio_path,text\n")
            for item in segments:
                if "who" not in item:
                    item["who"] = "unknown"
                f.write("%s,%f,%f,%f,%s, %s\n" % (item["who"],
                                                  item["start"],
                                                  item["end"],
                                                  item["end"] - item["start"],
                                                  src,
                                                  item["text"]))


def process_task(cc, task, stop_event):
    args = task["args"]

    src = args["src"]
    segment_file = args["segments"]
    vtt = args["vtt"]
    dst = args["dst"]  # JSON subtitle file
    speakers = dst.replace("_subs.json", "_speakers.json")
    cutoff = args.get("cutoff", 0.10)

    people_dir = args.get("people_dir", None)
    people_src = args.get("people", "")
    guess_people = args.get("guess_people", True)
    castsource = speakers.replace("_speakers.json", "_people.json")

    if not vtt.endswith("json"):
        import mod_reformat
        parser = mod_reformat.SubParser()
        print("Loading subtitles from '%s'" % vtt)
        subs = parser.load_srt(vtt)
    else:
        with open(vtt, "r") as f:
            subs = json.load(f)
    if len(subs) == 0:
        raise Exception("No subtitles in file '%s'" % vtt)
    cc.log.debug("Loaded %d subtitles" % len(subs))

    cc.status["progress"] = 0
    vc = VoiceCompare(cc.log)
    vc._load_model()

    if os.path.exists(dst) and os.path.getsize(dst) > 10:
        cc.log.warning("Cache didn't catch this one either")
        return 100, {"dst": dst, "cast": castsource}

    if segment_file.endswith(".csv"):
        segments = vc.read_csv(segment_file)
    else:
        raise Exception("Bad file format for segments: '%s'" % segment_file)

    cc.log.debug("Loaded %d segments" % len(segments))

    cc.status["progress"] = 1
    cc.status["state"] = "Loading embeddings"
    embeddings = vc.load_embeddings(src, segments)
    cc.status["progress"] = 5
    cc.log.debug("Loaded %d embeddings" % len(embeddings))
    # TRY THIS:
    # Input a list of people, with start-end for their voices
    # GO through the file and find best match
    known_people = None
    known_items = {}
    if os.path.exists(people_src) and os.path.getsize(people_src) > 10:
        cc.log.info("Loading candidate people from '%s'" % people_src)
        known_items = vc.load_person_list(people_src, people_dir)

        cc.log.info("Loaded %d candidates" % len(known_items))

    if len(known_items) == 0 or guess_people:  # not known_people:
        # Need to guess
        cc.status["state"] = "Auto-detecting people"
        # We could re-use the embeddings here and save us a LOT of time
        known_people = vc.guess_known_items(src, segments, embeddings)
        cc.log.debug("Got %d known items" % len(known_people))
        # DEBUGGING
        c = vc.people_to_cast(src, known_people)
        print("CAST")
        print(c)
        with open(castsource, "w") as f:
            json.dump(c, f, indent=" ")

        # TODO: Just strip off the timestamps from known_people
        known_items.update(vc.build_known_items(c))

    cc.status["progress"] = 15
    cc.status["status"] = "Matching"
    best_matches = vc.find_best_matches_known_items(known_items, embeddings)
    cc.status["progress"] = 55
    cc.status["state"] = "Building cast"
    cast = vc.build_cast_known_items(embeddings, known_items, best_matches, cutoff)
    cc.log.debug("%d cast memebers" % len(cast))
    # best_matches = vc.find_best_matches(embeddings)
    # cast = vc.build_cast(best_matches, cutoff)
    with open("/tmp/cast.json", "w") as f:
        json.dump(cast, f, indent=" ")
    # RESYNC
    # subs = vc.realign_subs(subs, segments, cast)

    # Go through the segments, find who is the most likely speaker, then update the thing
    cc.status["state"] = "Identify speakers"
    cc.status["progress"] = 75
    for s in subs:
        s["who"] = vc.find_most_likely_speaker(cast, s["start"], s["end"])

    # Also dump speakers for visual debugging
    dataset = []
    for person in cast:
        for t in cast[person]:
            t["who"] = person
            dataset.append(t)
    new_speak = []
    dataset.sort(key=operator.itemgetter("start"))
    for speaker in dataset:
        if len(new_speak) == 0:
            new_speak.append(speaker)
            continue

        if speaker["start"] == new_speak[-1]["end"] and speaker["who"] == new_speak[-1]["who"]:
            new_speak[-1]["end"] = speaker["end"]
        else:
            new_speak.append(speaker)
    
    with open(speakers, "w") as f:
        json.dump(new_speak, f, indent=" ")

    cc.status["state"] = "Realign subs"
    cc.status["progress"] = 80
    subs = vc.realign_subs_whisper(src, subs, new_speak, cast)

    # Write the CSV back
    # vc.write_csv(segments, dst)
    with open(dst, "w") as f:
        json.dump(subs, f, indent=" ")

    return 100, {"dst": dst, "cast": castsource}


if __name__ == "__main__":
    # Small tool to create embeddings for people based on a peoples file
    
    import sys
    with open(sys.argv[1], "r") as f:
        people = json.load(f)

    DBDIR="/home/njaal-local/peopleDB/"
    import pickle
    import base64

    vc = VoiceCompare(None)
    known_items = vc.build_known_items(people)

    for pid in people:
        person = people[pid]

        if "name" not in person:
            # likely extra colors only, don't worry
            continue

        print("PERSON", person)
        if not isinstance(person["name"], str):
            print("BAD PERSON", person["name"])
            continue

        name = str(person["name"]).replace(" ", "_")
        if "segments" in person:
            del person["segments"]


        person["voice"] = base64.b64encode(pickle.dumps(known_items[pid])).decode("ascii")

        with open(os.path.join(DBDIR, name) + ".json", "w") as f:
            json.dump(person, f)

        # with open(os.path.join(DBDIR, name) + ".embeddings", "wb") as f:
        #    pickle.dump(known_items[pid], f)
