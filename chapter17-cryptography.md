# Chapter 17: Cryptography Fundamentals

## Introduction

Cryptography is the science of protecting information through mathematical techniques that ensure confidentiality, integrity, authenticity, and non-repudiation. From ancient Caesar ciphers to modern quantum-resistant algorithms, cryptography has evolved to become the foundation of digital security. This chapter explores classical and modern cryptographic techniques, including symmetric and asymmetric encryption, hash functions, digital signatures, and cryptographic protocols that secure our digital communications.

## 17.1 Classical Cryptography

### Substitution Ciphers

```python
class CaesarCipher:
    def __init__(self, shift=3):
        self.shift = shift % 26
        
    def encrypt(self, plaintext):
        """Encrypt using Caesar cipher"""
        ciphertext = []
        
        for char in plaintext.upper():
            if char.isalpha():
                # Shift character
                shifted = (ord(char) - ord('A') + self.shift) % 26
                ciphertext.append(chr(shifted + ord('A')))
            else:
                ciphertext.append(char)
        
        return ''.join(ciphertext)
    
    def decrypt(self, ciphertext):
        """Decrypt Caesar cipher"""
        # Decryption is encryption with negative shift
        self.shift = -self.shift
        plaintext = self.encrypt(ciphertext)
        self.shift = -self.shift
        return plaintext
    
    def break_cipher(self, ciphertext):
        """Break Caesar cipher using frequency analysis"""
        # English letter frequency (approximate)
        english_freq = {
            'E': 12.70, 'T': 9.06, 'A': 8.17, 'O': 7.51,
            'I': 6.97, 'N': 6.75, 'S': 6.33, 'H': 6.09,
            'R': 5.99, 'D': 4.25, 'L': 4.03, 'C': 2.78,
            'U': 2.76, 'M': 2.41, 'W': 2.36, 'F': 2.23,
            'G': 2.02, 'Y': 1.97, 'P': 1.93, 'B': 1.29,
            'V': 0.98, 'K': 0.77, 'J': 0.15, 'X': 0.15,
            'Q': 0.10, 'Z': 0.07
        }
        
        best_shift = 0
        best_score = float('inf')
        
        for shift in range(26):
            # Try each possible shift
            test_cipher = CaesarCipher(shift)
            test_plain = test_cipher.decrypt(ciphertext)
            
            # Calculate frequency score
            freq = self.calculate_frequency(test_plain)
            score = self.chi_squared(freq, english_freq)
            
            if score < best_score:
                best_score = score
                best_shift = shift
        
        return best_shift
    
    def calculate_frequency(self, text):
        """Calculate letter frequency in text"""
        counts = {}
        total = 0
        
        for char in text.upper():
            if char.isalpha():
                counts[char] = counts.get(char, 0) + 1
                total += 1
        
        return {char: (count / total) * 100 
                for char, count in counts.items()}
    
    def chi_squared(self, observed, expected):
        """Calculate chi-squared statistic"""
        score = 0
        for char in expected:
            obs = observed.get(char, 0)
            exp = expected[char]
            score += ((obs - exp) ** 2) / exp
        return score

class VigenereCipher:
    def __init__(self, key):
        self.key = key.upper()
    
    def encrypt(self, plaintext):
        """Polyalphabetic substitution cipher"""
        ciphertext = []
        key_index = 0
        
        for char in plaintext.upper():
            if char.isalpha():
                # Get shift from key
                shift = ord(self.key[key_index % len(self.key)]) - ord('A')
                
                # Encrypt character
                encrypted = (ord(char) - ord('A') + shift) % 26
                ciphertext.append(chr(encrypted + ord('A')))
                
                key_index += 1
            else:
                ciphertext.append(char)
        
        return ''.join(ciphertext)
    
    def decrypt(self, ciphertext):
        """Decrypt Vigenere cipher"""
        plaintext = []
        key_index = 0
        
        for char in ciphertext.upper():
            if char.isalpha():
                # Get shift from key
                shift = ord(self.key[key_index % len(self.key)]) - ord('A')
                
                # Decrypt character
                decrypted = (ord(char) - ord('A') - shift) % 26
                plaintext.append(chr(decrypted + ord('A')))
                
                key_index += 1
            else:
                plaintext.append(char)
        
        return ''.join(plaintext)

class PlayfairCipher:
    def __init__(self, key):
        self.key = key.upper().replace('J', 'I')
        self.matrix = self.create_matrix()
    
    def create_matrix(self):
        """Create 5x5 Playfair matrix"""
        # Remove duplicates from key
        seen = set()
        key_letters = []
        for char in self.key:
            if char not in seen and char.isalpha():
                seen.add(char)
                key_letters.append(char)
        
        # Add remaining letters
        for char in 'ABCDEFGHIKLMNOPQRSTUVWXYZ':
            if char not in seen:
                key_letters.append(char)
        
        # Create 5x5 matrix
        matrix = []
        for i in range(5):
            matrix.append(key_letters[i*5:(i+1)*5])
        
        return matrix
    
    def find_position(self, char):
        """Find character position in matrix"""
        for i in range(5):
            for j in range(5):
                if self.matrix[i][j] == char:
                    return i, j
        return None
    
    def encrypt_pair(self, a, b):
        """Encrypt a pair of characters"""
        row_a, col_a = self.find_position(a)
        row_b, col_b = self.find_position(b)
        
        if row_a == row_b:
            # Same row - shift right
            return (self.matrix[row_a][(col_a + 1) % 5],
                   self.matrix[row_b][(col_b + 1) % 5])
        elif col_a == col_b:
            # Same column - shift down
            return (self.matrix[(row_a + 1) % 5][col_a],
                   self.matrix[(row_b + 1) % 5][col_b])
        else:
            # Rectangle - swap columns
            return (self.matrix[row_a][col_b],
                   self.matrix[row_b][col_a])
```

### Transposition Ciphers

```python
class RailFenceCipher:
    def __init__(self, rails):
        self.rails = rails
    
    def encrypt(self, plaintext):
        """Encrypt using rail fence transposition"""
        # Remove spaces
        plaintext = plaintext.replace(' ', '')
        
        # Create rail fence pattern
        fence = [[] for _ in range(self.rails)]
        rail = 0
        direction = 1
        
        for char in plaintext:
            fence[rail].append(char)
            rail += direction
            
            if rail == self.rails - 1 or rail == 0:
                direction = -direction
        
        # Read off ciphertext
        ciphertext = ''.join(''.join(rail) for rail in fence)
        return ciphertext
    
    def decrypt(self, ciphertext):
        """Decrypt rail fence cipher"""
        # Calculate fence pattern
        fence = [['' for _ in range(len(ciphertext))] 
                for _ in range(self.rails)]
        
        # Mark positions
        rail = 0
        direction = 1
        for i in range(len(ciphertext)):
            fence[rail][i] = '*'
            rail += direction
            
            if rail == self.rails - 1 or rail == 0:
                direction = -direction
        
        # Fill in ciphertext
        index = 0
        for r in range(self.rails):
            for c in range(len(ciphertext)):
                if fence[r][c] == '*':
                    fence[r][c] = ciphertext[index]
                    index += 1
        
        # Read off plaintext
        plaintext = []
        rail = 0
        direction = 1
        for i in range(len(ciphertext)):
            plaintext.append(fence[rail][i])
            rail += direction
            
            if rail == self.rails - 1 or rail == 0:
                direction = -direction
        
        return ''.join(plaintext)

class ColumnarTransposition:
    def __init__(self, key):
        self.key = key
        self.order = self.get_column_order()
    
    def get_column_order(self):
        """Get column read order from key"""
        indexed_key = [(char, i) for i, char in enumerate(self.key)]
        sorted_key = sorted(indexed_key)
        return [i for _, i in sorted_key]
    
    def encrypt(self, plaintext):
        """Encrypt using columnar transposition"""
        # Pad plaintext if necessary
        while len(plaintext) % len(self.key) != 0:
            plaintext += 'X'
        
        # Write into columns
        columns = [[] for _ in range(len(self.key))]
        for i, char in enumerate(plaintext):
            columns[i % len(self.key)].append(char)
        
        # Read columns in key order
        ciphertext = []
        for i in self.order:
            ciphertext.extend(columns[i])
        
        return ''.join(ciphertext)
```

## 17.2 Symmetric Cryptography

### Block Ciphers

```python
class SimpleDES:
    """Simplified DES for educational purposes"""
    
    def __init__(self, key):
        self.key = key
        self.subkeys = self.generate_subkeys()
    
    def generate_subkeys(self):
        """Generate round subkeys from main key"""
        # Simplified key schedule
        subkeys = []
        for i in range(16):
            # Rotate and select bits
            rotated = self.rotate_left(self.key, i + 1)
            subkey = self.permute(rotated, self.PC2)
            subkeys.append(subkey)
        return subkeys
    
    def initial_permutation(self, block):
        """Apply initial permutation"""
        IP = [58, 50, 42, 34, 26, 18, 10, 2,
              60, 52, 44, 36, 28, 20, 12, 4,
              62, 54, 46, 38, 30, 22, 14, 6,
              64, 56, 48, 40, 32, 24, 16, 8,
              57, 49, 41, 33, 25, 17, 9, 1,
              59, 51, 43, 35, 27, 19, 11, 3,
              61, 53, 45, 37, 29, 21, 13, 5,
              63, 55, 47, 39, 31, 23, 15, 7]
        
        return self.permute(block, IP)
    
    def feistel_round(self, left, right, subkey):
        """One round of Feistel network"""
        # Expansion
        expanded = self.expand(right)
        
        # XOR with subkey
        xored = self.xor(expanded, subkey)
        
        # S-box substitution
        substituted = self.sbox_substitution(xored)
        
        # Permutation
        permuted = self.permute(substituted, self.P)
        
        # XOR with left half
        new_right = self.xor(left, permuted)
        
        return right, new_right
    
    def encrypt_block(self, block):
        """Encrypt 64-bit block"""
        # Initial permutation
        permuted = self.initial_permutation(block)
        
        # Split into halves
        left = permuted[:32]
        right = permuted[32:]
        
        # 16 rounds
        for i in range(16):
            left, right = self.feistel_round(left, right, self.subkeys[i])
        
        # Final swap
        combined = right + left
        
        # Final permutation
        encrypted = self.final_permutation(combined)
        
        return encrypted

class AES:
    """Advanced Encryption Standard implementation"""
    
    def __init__(self, key):
        self.key = key
        self.round_keys = self.key_expansion()
        
        # S-box for SubBytes
        self.sbox = [
            0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5,
            # ... (256 bytes total)
        ]
        
    def key_expansion(self):
        """Expand key into round keys"""
        # AES-128: 10 rounds, AES-192: 12 rounds, AES-256: 14 rounds
        key_size = len(self.key)
        
        if key_size == 16:
            rounds = 10
        elif key_size == 24:
            rounds = 12
        elif key_size == 32:
            rounds = 14
        else:
            raise ValueError("Invalid key size")
        
        # Key schedule generation
        expanded = list(self.key)
        
        for i in range(rounds):
            # Complex key expansion logic
            pass
        
        return expanded
    
    def sub_bytes(self, state):
        """Substitute bytes using S-box"""
        for i in range(16):
            state[i] = self.sbox[state[i]]
        return state
    
    def shift_rows(self, state):
        """Shift rows of state matrix"""
        # Convert to 4x4 matrix
        matrix = [state[i:i+4] for i in range(0, 16, 4)]
        
        # Shift each row
        for i in range(4):
            matrix[i] = matrix[i][i:] + matrix[i][:i]
        
        # Convert back to list
        return [byte for row in matrix for byte in row]
    
    def mix_columns(self, state):
        """Mix columns using Galois field multiplication"""
        def galois_mult(a, b):
            """Multiplication in GF(2^8)"""
            p = 0
            for _ in range(8):
                if b & 1:
                    p ^= a
                hi_bit = a & 0x80
                a = (a << 1) & 0xFF
                if hi_bit:
                    a ^= 0x1B  # Reduction polynomial
                b >>= 1
            return p
        
        # Mix each column
        for i in range(4):
            col = state[i::4]
            state[i] = galois_mult(2, col[0]) ^ galois_mult(3, col[1]) ^ col[2] ^ col[3]
            state[i+4] = col[0] ^ galois_mult(2, col[1]) ^ galois_mult(3, col[2]) ^ col[3]
            state[i+8] = col[0] ^ col[1] ^ galois_mult(2, col[2]) ^ galois_mult(3, col[3])
            state[i+12] = galois_mult(3, col[0]) ^ col[1] ^ col[2] ^ galois_mult(2, col[3])
        
        return state
    
    def add_round_key(self, state, round_key):
        """XOR state with round key"""
        for i in range(16):
            state[i] ^= round_key[i]
        return state
    
    def encrypt_block(self, plaintext):
        """Encrypt 128-bit block"""
        state = list(plaintext)
        
        # Initial round
        state = self.add_round_key(state, self.round_keys[0])
        
        # Main rounds
        for round in range(1, len(self.round_keys) - 1):
            state = self.sub_bytes(state)
            state = self.shift_rows(state)
            state = self.mix_columns(state)
            state = self.add_round_key(state, self.round_keys[round])
        
        # Final round (no MixColumns)
        state = self.sub_bytes(state)
        state = self.shift_rows(state)
        state = self.add_round_key(state, self.round_keys[-1])
        
        return bytes(state)
```

### Stream Ciphers

```python
class RC4:
    def __init__(self, key):
        self.key = key
        self.S = self.ksa()
    
    def ksa(self):
        """Key Scheduling Algorithm"""
        S = list(range(256))
        j = 0
        
        for i in range(256):
            j = (j + S[i] + self.key[i % len(self.key)]) % 256
            S[i], S[j] = S[j], S[i]
        
        return S
    
    def prga(self):
        """Pseudo-Random Generation Algorithm"""
        S = self.S.copy()
        i = j = 0
        
        while True:
            i = (i + 1) % 256
            j = (j + S[i]) % 256
            S[i], S[j] = S[j], S[i]
            
            K = S[(S[i] + S[j]) % 256]
            yield K
    
    def encrypt(self, plaintext):
        """Encrypt using RC4 stream cipher"""
        keystream = self.prga()
        ciphertext = []
        
        for byte in plaintext:
            ciphertext.append(byte ^ next(keystream))
        
        return bytes(ciphertext)
    
    def decrypt(self, ciphertext):
        """Decryption is same as encryption for stream ciphers"""
        return self.encrypt(ciphertext)

class ChaCha20:
    def __init__(self, key, nonce):
        self.key = key
        self.nonce = nonce
        self.counter = 0
    
    def quarter_round(self, a, b, c, d):
        """ChaCha quarter round function"""
        def rotate_left(val, shift):
            return ((val << shift) | (val >> (32 - shift))) & 0xFFFFFFFF
        
        a = (a + b) & 0xFFFFFFFF
        d ^= a
        d = rotate_left(d, 16)
        
        c = (c + d) & 0xFFFFFFFF
        b ^= c
        b = rotate_left(b, 12)
        
        a = (a + b) & 0xFFFFFFFF
        d ^= a
        d = rotate_left(d, 8)
        
        c = (c + d) & 0xFFFFFFFF
        b ^= c
        b = rotate_left(b, 7)
        
        return a, b, c, d
    
    def chacha_block(self):
        """Generate ChaCha20 keystream block"""
        # Initialize state
        state = [
            0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,  # Constants
            *self.key,  # 8 words of key
            self.counter,  # Counter
            *self.nonce  # 3 words of nonce
        ]
        
        working_state = state.copy()
        
        # 20 rounds (10 double rounds)
        for _ in range(10):
            # Column rounds
            working_state[0], working_state[4], working_state[8], working_state[12] = \
                self.quarter_round(working_state[0], working_state[4], 
                                 working_state[8], working_state[12])
            # ... more quarter rounds
            
            # Diagonal rounds
            # ... diagonal quarter rounds
        
        # Add original state
        for i in range(16):
            working_state[i] = (working_state[i] + state[i]) & 0xFFFFFFFF
        
        self.counter += 1
        return working_state
```

## 17.3 Asymmetric Cryptography

### RSA Algorithm

```python
class RSA:
    def __init__(self, bits=2048):
        self.bits = bits
        self.public_key, self.private_key = self.generate_keypair()
    
    def generate_prime(self, bits):
        """Generate a prime number of specified bit length"""
        import random
        
        while True:
            # Generate random odd number
            n = random.getrandbits(bits)
            n |= (1 << bits - 1) | 1  # Set MSB and LSB
            
            if self.miller_rabin(n):
                return n
    
    def miller_rabin(self, n, k=5):
        """Miller-Rabin primality test"""
        if n < 2:
            return False
        
        # Write n-1 as 2^r * d
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        
        # Witness loop
        import random
        for _ in range(k):
            a = random.randrange(2, n - 1)
            x = pow(a, d, n)
            
            if x == 1 or x == n - 1:
                continue
            
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        
        return True
    
    def extended_gcd(self, a, b):
        """Extended Euclidean algorithm"""
        if a == 0:
            return b, 0, 1
        
        gcd, x1, y1 = self.extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        
        return gcd, x, y
    
    def mod_inverse(self, a, m):
        """Modular multiplicative inverse"""
        gcd, x, _ = self.extended_gcd(a, m)
        
        if gcd != 1:
            raise ValueError("Modular inverse does not exist")
        
        return (x % m + m) % m
    
    def generate_keypair(self):
        """Generate RSA public/private key pair"""
        # Generate two large primes
        p = self.generate_prime(self.bits // 2)
        q = self.generate_prime(self.bits // 2)
        
        # Calculate modulus
        n = p * q
        
        # Calculate Euler's totient
        phi = (p - 1) * (q - 1)
        
        # Choose public exponent (commonly 65537)
        e = 65537
        
        # Calculate private exponent
        d = self.mod_inverse(e, phi)
        
        # Return (e, n) as public key, (d, n) as private key
        return (e, n), (d, n)
    
    def encrypt(self, message, public_key):
        """Encrypt message using public key"""
        e, n = public_key
        
        # Convert message to integer
        m = int.from_bytes(message.encode(), 'big')
        
        if m >= n:
            raise ValueError("Message too large")
        
        # Encrypt: c = m^e mod n
        c = pow(m, e, n)
        
        return c
    
    def decrypt(self, ciphertext, private_key):
        """Decrypt ciphertext using private key"""
        d, n = private_key
        
        # Decrypt: m = c^d mod n
        m = pow(ciphertext, d, n)
        
        # Convert back to bytes
        byte_length = (m.bit_length() + 7) // 8
        message_bytes = m.to_bytes(byte_length, 'big')
        
        return message_bytes.decode()
    
    def sign(self, message, private_key):
        """Create digital signature"""
        d, n = private_key
        
        # Hash message
        import hashlib
        hash_value = hashlib.sha256(message.encode()).digest()
        hash_int = int.from_bytes(hash_value, 'big')
        
        # Sign: s = h^d mod n
        signature = pow(hash_int, d, n)
        
        return signature
    
    def verify(self, message, signature, public_key):
        """Verify digital signature"""
        e, n = public_key
        
        # Hash message
        import hashlib
        hash_value = hashlib.sha256(message.encode()).digest()
        hash_int = int.from_bytes(hash_value, 'big')
        
        # Verify: h' = s^e mod n
        hash_from_sig = pow(signature, e, n)
        
        return hash_int == hash_from_sig
```

### Elliptic Curve Cryptography

```python
class EllipticCurve:
    def __init__(self, a, b, p):
        """Initialize elliptic curve y^2 = x^3 + ax + b (mod p)"""
        self.a = a
        self.b = b
        self.p = p
    
    def is_on_curve(self, point):
        """Check if point is on the curve"""
        if point is None:
            return True  # Point at infinity
        
        x, y = point
        return (y * y) % self.p == (x * x * x + self.a * x + self.b) % self.p
    
    def point_add(self, P, Q):
        """Add two points on the elliptic curve"""
        if P is None:
            return Q
        if Q is None:
            return P
        
        x1, y1 = P
        x2, y2 = Q
        
        if x1 == x2:
            if y1 == y2:
                # Point doubling
                s = ((3 * x1 * x1 + self.a) * self.mod_inverse(2 * y1, self.p)) % self.p
            else:
                # Points are inverses
                return None
        else:
            # Point addition
            s = ((y2 - y1) * self.mod_inverse(x2 - x1, self.p)) % self.p
        
        x3 = (s * s - x1 - x2) % self.p
        y3 = (s * (x1 - x3) - y1) % self.p
        
        return (x3, y3)
    
    def scalar_mult(self, k, P):
        """Multiply point by scalar using double-and-add"""
        if k == 0:
            return None
        if k == 1:
            return P
        
        result = None
        addend = P
        
        while k:
            if k & 1:
                result = self.point_add(result, addend)
            
            addend = self.point_add(addend, addend)
            k >>= 1
        
        return result
    
    def mod_inverse(self, a, m):
        """Modular inverse using extended Euclidean algorithm"""
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y
        
        gcd, x, _ = extended_gcd(a, m)
        if gcd != 1:
            raise ValueError("Modular inverse does not exist")
        return (x % m + m) % m

class ECDH:
    """Elliptic Curve Diffie-Hellman"""
    
    def __init__(self, curve, generator):
        self.curve = curve
        self.G = generator  # Base point
        self.private_key = None
        self.public_key = None
    
    def generate_keypair(self):
        """Generate ECDH key pair"""
        import random
        
        # Generate private key (random scalar)
        self.private_key = random.randrange(1, self.curve.p)
        
        # Calculate public key (private_key * G)
        self.public_key = self.curve.scalar_mult(self.private_key, self.G)
        
        return self.public_key
    
    def compute_shared_secret(self, other_public_key):
        """Compute shared secret from other party's public key"""
        if self.private_key is None:
            raise ValueError("Private key not generated")
        
        # Shared secret = private_key * other_public_key
        shared_point = self.curve.scalar_mult(self.private_key, other_public_key)
        
        # Use x-coordinate as shared secret
        return shared_point[0] if shared_point else 0

class ECDSA:
    """Elliptic Curve Digital Signature Algorithm"""
    
    def __init__(self, curve, generator, order):
        self.curve = curve
        self.G = generator
        self.n = order  # Order of generator
    
    def sign(self, message, private_key):
        """Create ECDSA signature"""
        import hashlib
        import random
        
        # Hash message
        hash_value = hashlib.sha256(message.encode()).digest()
        z = int.from_bytes(hash_value, 'big')
        
        while True:
            # Generate random k
            k = random.randrange(1, self.n)
            
            # Calculate r = (k * G).x mod n
            point = self.curve.scalar_mult(k, self.G)
            r = point[0] % self.n
            
            if r == 0:
                continue
            
            # Calculate s = k^(-1) * (z + r * private_key) mod n
            k_inv = self.curve.mod_inverse(k, self.n)
            s = (k_inv * (z + r * private_key)) % self.n
            
            if s != 0:
                return (r, s)
    
    def verify(self, message, signature, public_key):
        """Verify ECDSA signature"""
        import hashlib
        
        r, s = signature
        
        # Check signature values
        if not (1 <= r < self.n and 1 <= s < self.n):
            return False
        
        # Hash message
        hash_value = hashlib.sha256(message.encode()).digest()
        z = int.from_bytes(hash_value, 'big')
        
        # Calculate s^(-1) mod n
        s_inv = self.curve.mod_inverse(s, self.n)
        
        # Calculate u1 = z * s^(-1) mod n and u2 = r * s^(-1) mod n
        u1 = (z * s_inv) % self.n
        u2 = (r * s_inv) % self.n
        
        # Calculate point (x, y) = u1 * G + u2 * public_key
        point1 = self.curve.scalar_mult(u1, self.G)
        point2 = self.curve.scalar_mult(u2, public_key)
        point = self.curve.point_add(point1, point2)
        
        if point is None:
            return False
        
        # Verify r = x mod n
        return r == point[0] % self.n
```

## 17.4 Hash Functions

### Cryptographic Hash Functions

```python
class SHA256:
    def __init__(self):
        # Initial hash values (first 32 bits of fractional parts of square roots of first 8 primes)
        self.h = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
        ]
        
        # Round constants (first 32 bits of fractional parts of cube roots of first 64 primes)
        self.k = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
            # ... (64 constants total)
        ]
    
    def right_rotate(self, n, b):
        """Right rotate n by b bits"""
        return ((n >> b) | (n << (32 - b))) & 0xFFFFFFFF
    
    def pad_message(self, message):
        """Pad message to multiple of 512 bits"""
        msg_len = len(message)
        message += b'\x80'  # Append bit '1'
        
        # Pad with zeros
        while (len(message) + 8) % 64 != 0:
            message += b'\x00'
        
        # Append original length as 64-bit big-endian
        message += (msg_len * 8).to_bytes(8, 'big')
        
        return message
    
    def process_chunk(self, chunk):
        """Process 512-bit chunk"""
        # Break chunk into 16 32-bit words
        w = []
        for i in range(16):
            w.append(int.from_bytes(chunk[i*4:(i+1)*4], 'big'))
        
        # Extend to 64 words
        for i in range(16, 64):
            s0 = self.right_rotate(w[i-15], 7) ^ self.right_rotate(w[i-15], 18) ^ (w[i-15] >> 3)
            s1 = self.right_rotate(w[i-2], 17) ^ self.right_rotate(w[i-2], 19) ^ (w[i-2] >> 10)
            w.append((w[i-16] + s0 + w[i-7] + s1) & 0xFFFFFFFF)
        
        # Initialize working variables
        a, b, c, d, e, f, g, h = self.h
        
        # Main loop
        for i in range(64):
            S1 = self.right_rotate(e, 6) ^ self.right_rotate(e, 11) ^ self.right_rotate(e, 25)
            ch = (e & f) ^ (~e & g)
            temp1 = (h + S1 + ch + self.k[i] + w[i]) & 0xFFFFFFFF
            
            S0 = self.right_rotate(a, 2) ^ self.right_rotate(a, 13) ^ self.right_rotate(a, 22)
            maj = (a & b) ^ (a & c) ^ (b & c)
            temp2 = (S0 + maj) & 0xFFFFFFFF
            
            h, g, f, e = g, f, e, (d + temp1) & 0xFFFFFFFF
            d, c, b, a = c, b, a, (temp1 + temp2) & 0xFFFFFFFF
        
        # Update hash values
        self.h[0] = (self.h[0] + a) & 0xFFFFFFFF
        self.h[1] = (self.h[1] + b) & 0xFFFFFFFF
        self.h[2] = (self.h[2] + c) & 0xFFFFFFFF
        self.h[3] = (self.h[3] + d) & 0xFFFFFFFF
        self.h[4] = (self.h[4] + e) & 0xFFFFFFFF
        self.h[5] = (self.h[5] + f) & 0xFFFFFFFF
        self.h[6] = (self.h[6] + g) & 0xFFFFFFFF
        self.h[7] = (self.h[7] + h) & 0xFFFFFFFF
    
    def hash(self, message):
        """Compute SHA-256 hash"""
        # Pad message
        padded = self.pad_message(message)
        
        # Process each 512-bit chunk
        for i in range(0, len(padded), 64):
            self.process_chunk(padded[i:i+64])
        
        # Produce final hash value
        digest = b''
        for h in self.h:
            digest += h.to_bytes(4, 'big')
        
        return digest

class HMAC:
    """Hash-based Message Authentication Code"""
    
    def __init__(self, key, hash_func=SHA256):
        self.key = key
        self.hash_func = hash_func
        self.block_size = 64  # SHA-256 block size
        
        # Pad or hash key to block size
        if len(key) > self.block_size:
            self.key = hash_func().hash(key)
        if len(key) < self.block_size:
            self.key = key + b'\x00' * (self.block_size - len(key))
    
    def compute(self, message):
        """Compute HMAC"""
        # Create padded keys
        opad = bytes(0x5C ^ b for b in self.key)
        ipad = bytes(0x36 ^ b for b in self.key)
        
        # HMAC = H(K ⊕ opad || H(K ⊕ ipad || message))
        inner_hash = self.hash_func().hash(ipad + message)
        outer_hash = self.hash_func().hash(opad + inner_hash)
        
        return outer_hash
    
    def verify(self, message, mac):
        """Verify HMAC"""
        computed_mac = self.compute(message)
        
        # Constant-time comparison to prevent timing attacks
        return self.constant_time_compare(computed_mac, mac)
    
    def constant_time_compare(self, a, b):
        """Compare two byte strings in constant time"""
        if len(a) != len(b):
            return False
        
        result = 0
        for x, y in zip(a, b):
            result |= x ^ y
        
        return result == 0
```

## 17.5 Cryptographic Protocols

### Diffie-Hellman Key Exchange

```python
class DiffieHellman:
    def __init__(self, p=None, g=None):
        if p is None:
            # Use standard DH parameters (2048-bit MODP group)
            self.p = int('FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1'
                        '29024E088A67CC74020BBEA63B139B22514A08798E3404DD'
                        'EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245'
                        'E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED'
                        'EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3D'
                        'C2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F'
                        '83655D23DCA3AD961C62F356208552BB9ED529077096966D'
                        '670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B'
                        'E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9'
                        'DE2BCBF6955817183995497CEA956AE515D2261898FA0510'
                        '15728E5A8AACAA68FFFFFFFFFFFFFFFF', 16)
            self.g = 2
        else:
            self.p = p
            self.g = g
        
        self.private_key = None
        self.public_key = None
    
    def generate_keys(self):
        """Generate DH key pair"""
        import random
        
        # Generate private key
        self.private_key = random.randrange(2, self.p - 1)
        
        # Calculate public key: g^private mod p
        self.public_key = pow(self.g, self.private_key, self.p)
        
        return self.public_key
    
    def compute_shared_secret(self, other_public_key):
        """Compute shared secret"""
        if self.private_key is None:
            raise ValueError("Private key not generated")
        
        # Shared secret: other_public^private mod p
        shared_secret = pow(other_public_key, self.private_key, self.p)
        
        return shared_secret
    
    def derive_key(self, shared_secret, info=b''):
        """Derive symmetric key from shared secret using KDF"""
        import hashlib
        
        # Simple KDF using SHA-256
        key_material = shared_secret.to_bytes((shared_secret.bit_length() + 7) // 8, 'big')
        derived = hashlib.sha256(key_material + info).digest()
        
        return derived

class TLS_Handshake:
    """Simplified TLS handshake protocol"""
    
    def __init__(self):
        self.version = "TLS 1.3"
        self.cipher_suites = [
            'TLS_AES_128_GCM_SHA256',
            'TLS_AES_256_GCM_SHA384',
            'TLS_CHACHA20_POLY1305_SHA256'
        ]
    
    def client_hello(self):
        """Generate ClientHello message"""
        import random
        
        client_hello = {
            'version': self.version,
            'random': random.randbytes(32),
            'session_id': b'',
            'cipher_suites': self.cipher_suites,
            'extensions': {
                'supported_groups': ['x25519', 'secp256r1'],
                'key_share': self.generate_key_shares()
            }
        }
        
        return client_hello
    
    def server_hello(self, client_hello):
        """Generate ServerHello message"""
        import random
        
        # Select cipher suite
        selected_cipher = client_hello['cipher_suites'][0]
        
        server_hello = {
            'version': self.version,
            'random': random.randbytes(32),
            'session_id': client_hello['session_id'],
            'cipher_suite': selected_cipher,
            'extensions': {
                'key_share': self.select_key_share(client_hello['extensions']['key_share'])
            }
        }
        
        return server_hello
    
    def derive_keys(self, shared_secret, handshake_messages):
        """Derive session keys from shared secret"""
        import hashlib
        import hmac
        
        # Simplified key derivation
        early_secret = hmac.new(b'\x00' * 32, shared_secret, hashlib.sha256).digest()
        
        handshake_hash = hashlib.sha256(b''.join(handshake_messages)).digest()
        
        derived_secret = hmac.new(early_secret, handshake_hash, hashlib.sha256).digest()
        
        # Derive specific keys
        client_key = hmac.new(derived_secret, b'client application key', hashlib.sha256).digest()[:16]
        server_key = hmac.new(derived_secret, b'server application key', hashlib.sha256).digest()[:16]
        
        return {
            'client_write_key': client_key,
            'server_write_key': server_key
        }
```

## 17.6 Applied Cryptography

### Password Security

```python
class PasswordHasher:
    def pbkdf2(self, password, salt, iterations=100000, key_length=32):
        """PBKDF2 key derivation function"""
        import hashlib
        import hmac
        
        def f(password, salt, iterations, block_num):
            """PBKDF2 inner function"""
            u = hmac.new(password, salt + block_num.to_bytes(4, 'big'), hashlib.sha256).digest()
            result = u
            
            for _ in range(iterations - 1):
                u = hmac.new(password, u, hashlib.sha256).digest()
                result = bytes(a ^ b for a, b in zip(result, u))
            
            return result
        
        # Derive key
        dk = b''
        block_num = 1
        
        while len(dk) < key_length:
            dk += f(password.encode(), salt, iterations, block_num)
            block_num += 1
        
        return dk[:key_length]
    
    def scrypt(self, password, salt, N=16384, r=8, p=1, key_length=32):
        """Scrypt memory-hard key derivation function"""
        # Simplified implementation
        import hashlib
        
        # Initial PBKDF2
        b = self.pbkdf2(password, salt, 1, p * 128 * r)
        
        # Scrypt core (simplified)
        v = []
        for i in range(N):
            v.append(hashlib.sha256(b).digest())
            b = hashlib.sha256(b + v[-1]).digest()
        
        # Final PBKDF2
        return self.pbkdf2(password, b, 1, key_length)
    
    def argon2(self, password, salt, time_cost=3, memory_cost=4096, parallelism=1):
        """Argon2 password hashing (simplified)"""
        import hashlib
        
        # Initialize memory blocks
        memory = [[0] * 1024 for _ in range(memory_cost)]
        
        # Initial hashing
        initial = hashlib.blake2b(
            password.encode() + salt + 
            time_cost.to_bytes(4, 'little') + 
            memory_cost.to_bytes(4, 'little')
        ).digest()
        
        # Fill memory (simplified)
        for t in range(time_cost):
            for i in range(memory_cost):
                if i == 0:
                    memory[i] = [int.from_bytes(initial[j:j+8], 'little') 
                                for j in range(0, 64, 8)]
                else:
                    # Mix with previous blocks
                    prev = memory[i-1]
                    memory[i] = [a ^ b for a, b in zip(memory[i], prev)]
        
        # Final hash
        final = b''.join(x.to_bytes(8, 'little') for x in memory[-1])
        return hashlib.blake2b(final, digest_size=32).digest()

class SecureRandom:
    """Cryptographically secure random number generation"""
    
    def __init__(self):
        import os
        self.entropy_pool = bytearray(os.urandom(256))
        self.counter = 0
    
    def get_random_bytes(self, n):
        """Generate n random bytes"""
        import hashlib
        
        result = bytearray()
        
        while len(result) < n:
            # Mix counter with entropy pool
            self.counter += 1
            seed = self.entropy_pool + self.counter.to_bytes(8, 'big')
            
            # Generate random bytes using hash
            random_block = hashlib.sha256(seed).digest()
            result.extend(random_block)
            
            # Update entropy pool
            self.entropy_pool = bytearray(hashlib.sha256(self.entropy_pool + random_block).digest())
        
        return bytes(result[:n])
    
    def get_random_int(self, min_val, max_val):
        """Generate random integer in range"""
        range_size = max_val - min_val + 1
        num_bytes = (range_size.bit_length() + 7) // 8
        
        while True:
            random_bytes = self.get_random_bytes(num_bytes)
            random_int = int.from_bytes(random_bytes, 'big')
            
            if random_int < range_size:
                return min_val + random_int
```

## Exercises

1. Implement a complete AES encryption system with:
   - All key sizes (128, 192, 256)
   - Multiple modes (ECB, CBC, CTR, GCM)
   - Proper padding schemes

2. Build an RSA cryptosystem that:
   - Generates secure keys
   - Implements OAEP padding
   - Provides digital signatures

3. Create an elliptic curve library with:
   - Multiple standard curves
   - Point arithmetic operations
   - ECDH and ECDSA implementations

4. Implement a TLS 1.3 handshake that:
   - Negotiates cipher suites
   - Performs key exchange
   - Derives session keys

5. Build a password manager with:
   - Secure key derivation
   - Encrypted storage
   - Zero-knowledge architecture

6. Create a blockchain proof-of-work system using:
   - SHA-256 hashing
   - Difficulty adjustment
   - Block validation

7. Implement a secure messaging protocol with:
   - End-to-end encryption
   - Perfect forward secrecy
   - Message authentication

8. Build a cryptographic toolkit that:
   - Analyzes cipher strength
   - Performs cryptanalysis
   - Generates secure random numbers

9. Create a digital signature system with:
   - Multiple algorithms (RSA, DSA, ECDSA)
   - Certificate generation
   - Signature verification

10. Implement a homomorphic encryption scheme that:
    - Allows computation on encrypted data
    - Preserves privacy
    - Provides practical performance

## Summary

This chapter covered fundamental cryptographic concepts:

- Classical cryptography introduced substitution and transposition techniques
- Symmetric cryptography provides efficient encryption with shared keys
- Asymmetric cryptography enables secure communication without shared secrets
- Hash functions ensure data integrity and enable digital signatures
- Cryptographic protocols combine primitives for secure communication
- Applied cryptography addresses real-world security challenges
- Key management and random number generation are critical for security
- Modern cryptography relies on mathematical hardness assumptions

Cryptography is essential for information security, enabling confidential communication, data integrity, authentication, and non-repudiation in our digital world. Understanding these principles is crucial for implementing secure systems.