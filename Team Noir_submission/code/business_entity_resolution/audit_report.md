# Preprocessing Audit Report
- **Source File**: `train_source1.tsv`
- **Total Rows Audited**: `5,000`

## 1. High-Level Alteration Summary
| Field | Total Records | Altered Records | % Altered | Unchanged Records | % Unchanged |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **business_name** | 5,000 | 5,000 | 100.00% | 0 | 0.00% |
| **business_address** | 5,000 | 5,000 | 100.00% | 0 | 0.00% |
| **country** | 5,000 | 1,975 | 39.50% | 3,025 | 60.50% |

- **Postal / PIN Codes Extracted**: `349` (6.98% of rows)

## 2. Detailed Breakdown of Manipulations Applied

### A. `business_name` Transformations
| Manipulation Type | Rows Affected | % of Audited Rows |
| :--- | :--- | :--- |
| Case Lowered | 5,000 | 100.00% |
| Legal Suffix Standardized | 3,218 | 64.36% |
| Punctuation Cleaned | 1,041 | 20.82% |
| Ampersand Expanded | 264 | 5.28% |
| Slash Separated | 7 | 0.14% |

### B. `business_address` Transformations
| Manipulation Type | Rows Affected | % of Audited Rows |
| :--- | :--- | :--- |
| Case Lowered | 5,000 | 100.00% |
| Punctuation Cleaned | 5,000 | 100.00% |
| Roadway Standardized | 3,674 | 73.48% |
| Unit/Direction Standardized | 1,181 | 23.62% |
| Slash Separated | 645 | 12.90% |
| Postal Code Extracted | 349 | 6.98% |
| Ampersand Expanded | 38 | 0.76% |
| State Abbreviated | 35 | 0.70% |
| Unicode / Accent Normalized | 1 | 0.02% |

### C. `country` Transformations
| Manipulation Type | Rows Affected | % of Audited Rows |
| :--- | :--- | :--- |
| Unchanged | 3,025 | 60.50% |
| Country Standardized to ISO | 1,975 | 39.50% |

## 3. Concrete Before vs After Examples

#### Example 1 (`S1-925783039`)
- **Raw Name**: `Orelee's Barbershop`
  - **Clean Name**: `orelee s barbershop` *(Tags: Case Lowered, Punctuation Cleaned)*
- **Raw Address**: `1795 Westchester Drive, High Point, NC`
  - **Clean Address**: `1795 westchester dr high point nc` *(Tags: Case Lowered, Roadway Standardized, Punctuation Cleaned)*
  - **Postal Code**: `[None]`
- **Country**: `US` -> `US`

#### Example 2 (`S1-773889195`)
- **Raw Name**: `Prime Money`
  - **Clean Name**: `prime money` *(Tags: Case Lowered)*
- **Raw Address**: `17560 Ellis Road, Tahlequah, OK`
  - **Clean Address**: `17560 ellis rd tahlequah ok` *(Tags: Case Lowered, Roadway Standardized, Postal Code Extracted, Punctuation Cleaned)*
  - **Postal Code**: `17560`
- **Country**: `US` -> `US`

#### Example 3 (`S1-377745466`)
- **Raw Name**: `B+ Retail Inc`
  - **Clean Name**: `b retail inc` *(Tags: Case Lowered, Legal Suffix Standardized, Punctuation Cleaned)*
- **Raw Address**: `1712 Montebello Avenue, Phoenix, AZ`
  - **Clean Address**: `1712 montebello ave phoenix az` *(Tags: Case Lowered, Roadway Standardized, Punctuation Cleaned)*
  - **Postal Code**: `[None]`
- **Country**: `US` -> `US`

#### Example 4 (`S1-133037285`)
- **Raw Name**: `Christ Chapel`
  - **Clean Name**: `christ chapel` *(Tags: Case Lowered)*
- **Raw Address**: `2100 Cameron Drive, Unit APARTMENT G, Dundalk, MD`
  - **Clean Address**: `2100 cameron dr unit apt g dundalk md` *(Tags: Case Lowered, Roadway Standardized, Unit/Direction Standardized, Punctuation Cleaned)*
  - **Postal Code**: `[None]`
- **Country**: `US` -> `US`

#### Example 5 (`S1-755362802`)
- **Raw Name**: `Prabhav Business Center`
  - **Clean Name**: `prabhav business center` *(Tags: Case Lowered)*
- **Raw Address**: `797, Lake Town Block A, Kolkata, Howrah, West Bengal`
  - **Clean Address**: `797 lake town block a kolkata howrah w bengal` *(Tags: Case Lowered, Unit/Direction Standardized, Punctuation Cleaned)*
  - **Postal Code**: `[None]`
- **Country**: `India` -> `IN`

#### Example 6 (`S1-851869949`)
- **Raw Name**: `Custom Wealth Services LLC`
  - **Clean Name**: `custom wealth services llc` *(Tags: Case Lowered, Legal Suffix Standardized)*
- **Raw Address**: `OH, Columbus, 5559 Orville Avenue`
  - **Clean Address**: `oh columbus 5559 orville ave` *(Tags: Case Lowered, Roadway Standardized, Punctuation Cleaned)*
  - **Postal Code**: `[None]`
- **Country**: `US` -> `US`

#### Example 7 (`S1-785847572`)
- **Raw Name**: `Consulting Nyasa Nursing Private Limited`
  - **Clean Name**: `consulting nyasa nursing pvt ltd` *(Tags: Case Lowered, Legal Suffix Standardized)*
- **Raw Address**: `2505, Tower 1, Oakwood, Runwal Greens, Mulund Goreagon Link Road, Near Fortis Hospital, Bhandup West, Mumbai, Maharashtra`
  - **Clean Address**: `2505 tower 1 oakwood runwal greens mulund goreagon link rd near fortis hospital bhandup w mumbai maharashtra` *(Tags: Case Lowered, Roadway Standardized, Unit/Direction Standardized, Punctuation Cleaned)*
  - **Postal Code**: `[None]`
- **Country**: `India` -> `IN`

#### Example 8 (`S1-27541239`)
- **Raw Name**: `Nexus Anchor Rain`
  - **Clean Name**: `nexus anchor rain` *(Tags: Case Lowered)*
- **Raw Address**: `1111 Church Street, Unit 2007, Nashville, TN`
  - **Clean Address**: `1111 church st unit 2007 nashville tn` *(Tags: Case Lowered, Roadway Standardized, Unit/Direction Standardized, Punctuation Cleaned)*
  - **Postal Code**: `[None]`
- **Country**: `US` -> `US`

#### Example 9 (`S1-629417405`)
- **Raw Name**: `Moore Bitwise Inc`
  - **Clean Name**: `moore bitwise inc` *(Tags: Case Lowered, Legal Suffix Standardized)*
- **Raw Address**: `337 Oakland Avenue, Michigan City, IN`
  - **Clean Address**: `337 oakland ave michigan city in` *(Tags: Case Lowered, Roadway Standardized, Punctuation Cleaned)*
  - **Postal Code**: `[None]`
- **Country**: `US` -> `US`

#### Example 10 (`S1-22305073`)
- **Raw Name**: `Dermatology Green Medicine`
  - **Clean Name**: `dermatology green medicine` *(Tags: Case Lowered)*
- **Raw Address**: `294 Meadowcreek Drive, Unit Unit 2, Village Of Pewaukee, WI`
  - **Clean Address**: `294 meadowcreek dr unit unit 2 village of pewaukee wi` *(Tags: Case Lowered, Roadway Standardized, Unit/Direction Standardized, Punctuation Cleaned)*
  - **Postal Code**: `[None]`
- **Country**: `US` -> `US`