import json
from pathlib import Path
import tempfile
import unittest

import gprsv_oracle as oracle


class OracleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="oracle-unit-", dir=oracle.ROOT)
        self.root = Path(self.temp.name).resolve()
        self.assertEqual(self.root.parent, oracle.ROOT)
        self.addCleanup(self.temp.cleanup)

    def file(self, name, text):
        path = self.root / name
        path.write_text(text, encoding="utf-8", newline="\n")
        return path

    def test_small_independent_full_oracle_and_self_factors(self):
        self.assertEqual(oracle.selftest()["status"], "PASS_CPU_ONLY")

    def test_abc_abcd_and_newpgen_parsers(self):
        abc = self.file("twins_a.pfgw", "ABC $a*3!+1 & $a*3!-1 // Sieved to 3\n1\n3\n5\n")
        abcd = self.file("twins_d.pfgw", "ABCD $a*3!+1 & $a*3!-1  [1] // Sieved to 3\n2\n2\n")
        oracle.same_survivors(oracle.parse_terms(abc), oracle.parse_terms(abcd))
        self.assertEqual(oracle.parse_terms(abc).units, {(1,0),(3,0),(5,0)})
        independ = self.file("ind.pfgw", "ABC $a*3#$b // Sieved to 13\n1 +1\n1 -1\n2 +1\n")
        parsed = oracle.parse_terms(independ)
        self.assertEqual(parsed.spec, oracle.Spec("primorial",3))
        self.assertEqual(parsed.units, {(1,1),(1,-1),(2,1)})
        npg = self.file("test.npg", "17:T:0:10:3\n1 2\n3 2\n")
        self.assertEqual(oracle.parse_terms(npg).spec, oracle.Spec("bn",2,10))
        self.assertEqual(oracle.parse_terms(npg).units, {(1,0),(3,0)})

    def test_empty_abcd_needs_explicit_context(self):
        empty = self.file("empty.pfgw", "")
        with self.assertRaises(ValueError):
            oracle.parse_terms(empty)
        parsed = oracle.parse_terms(empty,(oracle.Spec("factorial",3),"twin"))
        self.assertEqual(parsed.units,set())
        self.assertIsNone(parsed.sieved_to)

    def test_malformed_or_duplicate_terms_rejected(self):
        for text in ("ABC $a*3!$b\n1 0\n", "ABC $a*3!$b\n1 +1\n1 +1\n",
                     "ABCD $a*3!+1 & $a*3!-1 [1]\n0\n", "17:T:0:10:3\n1 2\n3 3\n",
                     "ABC $a*3!+1 & $a*5!-1\n1\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                oracle.parse_terms(self.file("bad.pfgw",text))

    def test_different_first_found_factors_are_both_valid(self):
        spec = oracle.Spec("factorial",3)
        left = self.file("left.factors", "5 | 29*3!+1\n")
        right = self.file("right.factors", "7 | 29*3!+1\n")
        for path in (left,right):
            result = oracle.verify_factors([path],spec,mode="independent",initial={(29,1)},survivors=set(),pmin=3,pmax=17)
            self.assertEqual(result["records"],1)
        self.assertNotEqual(left.read_bytes(),right.read_bytes())

    def test_factor_identity_self_factor_composite_and_sign_rejected(self):
        spec = oracle.Spec("factorial",3)
        for text in ("5 | 1*3!-1\n", "7 | 1*3!+1\n", "5 | 29*3!-1\n",
                     "35 | 29*3!+1\n", "5 | 29*3#+1\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                oracle.verify_factors([self.file("bad.factors",text)],spec)

    def test_oracle_prime_boundary_and_forced_high_prime_removal(self):
        self.assertEqual(oracle.interval_primes((1<<32)-10,(1<<32)+20),[4294967291,4294967311])
        plan = oracle.plan_command()
        selected = [case for case in plan["cases"] if case["mode"] == "independent" and case["forced_factor"]]
        for case in selected:
            with self.subTest(case=case["label"]):
                f = case["forced_factor"]
                spec = oracle.Spec(case["type"],case["n"])
                expected, detail = oracle.expected_survivors(spec,"independent",{(f["k"],f["c"])},f["p"]-1,f["p"])
                self.assertEqual(expected.units,set())
                self.assertEqual(detail["primes_tested"],1)
                self.assertLessEqual(f["p"],oracle.PMAX)

    def test_oracle_resource_caps(self):
        with self.assertRaises(ValueError):
            oracle.interval_primes(1,oracle.MAX_SPAN+2)
        with self.assertRaises(ValueError):
            oracle.generated(oracle.Spec("factorial",31),"twin",1,oracle.MAX_K_SPAN+1)
        with self.assertRaises(ValueError):
            oracle.Spec("primorial",100).multiplier()


if __name__ == "__main__":
    unittest.main()
