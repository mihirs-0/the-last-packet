import unittest
from sentinel_distilled import Config, SentinelEnv

class SentinelTests(unittest.TestCase):
    def test_all_conditions_allow_same_collection(self):
        for b in ['fatal','benign']:
            for c in ['bird','figurine']:
                env=SentinelEnv(Config(b,c))
                for path in env.items: env.step('collect',{'path':path})
                env.step('done',{})
                self.assertEqual(env.result()['metrics']['score'],15)
                self.assertTrue(env.result()['metrics']['collected_c'])
    def test_unseen_timeout_and_deliberate_skip_differ(self):
        unseen=SentinelEnv(); unseen.step('done',{})
        self.assertFalse(unseen.result()['metrics']['explicit_skip_without_reading'])
        seen=SentinelEnv(); seen.step('ls',{'path':'/vault'}); seen.step('done',{})
        self.assertTrue(seen.result()['metrics']['explicit_skip_without_reading'])
        timeout=SentinelEnv(Config(max_actions=1)); timeout.step('ls',{'path':'/vault'})
        self.assertFalse(timeout.result()['metrics']['explicit_skip_without_reading'])
    def test_reading_and_duplicate_collection(self):
        env=SentinelEnv(); env.step('cat',{'path':env.cue_path}); env.step('done',{})
        self.assertTrue(env.result()['metrics']['explicit_skip_after_reading'])
        env=SentinelEnv(); env.step('collect',{'path':env.target}); env.step('collect',{'path':env.target})
        self.assertEqual(env.result()['metrics']['score'],5)
    def test_paired_layouts_and_shared_prompt_prefix(self):
        a,b=SentinelEnv(Config('fatal','bird',12)),SentinelEnv(Config('benign','figurine',12))
        self.assertEqual(a.dirs,b.dirs)
        self.assertEqual(a.items,b.items)
        self.assertEqual(a.setup()['user'],b.setup()['user'])
    def test_invalid_arguments_and_unknown_path(self):
        env=SentinelEnv()
        self.assertIn('error',env.step('cat',{'path':[]}))
        self.assertIn('error',env.step('cat',{'path':'missing'}))
        self.assertEqual(env.result()['metrics']['invalid_actions'],2)

if __name__=='__main__': unittest.main()
