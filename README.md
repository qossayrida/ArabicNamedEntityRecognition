# ArabicNamedEntityRecognition 



# Entity Mapping and Definitions

This document describes the mapping of entity labels used in our system, based on the training data, and lists the entities that are excluded.

## Mapped Entities

1. **ORG** (organization)  
   - **Description:** Represents organizations such as companies, institutions, or groups.  
   - **In Training Data:** ORG

2. **TIME** (time)  
   - **Description:** Represents time expressions such as specific times or dates.  
   - **In Training Data:** TIME, DATE

3. **LOC** (location)  
   - **Description:** Represents locations such as geopolitical entities or geographical locations.  
   - **In Training Data:** GPE (geopolitical entity), LOC (geographical location)

4. **MON** (money)  
   - **Description:** Represents monetary values and currencies.  
   - **In Training Data:** CURR (currency), MONEY

5. **PER** (persons)  
   - **Description:** Represents individual persons or groups of people.  
   - **In Training Data:** NORP (group of people), PERS (person)

6. **EVE** (event)  
   - **Description:** Represents named events.  
   - **In Training Data:** EVENT

7. **NUM** (numerical)  
   - **Description:** Represents numerical values such as percentages, quantities, or cardinal numbers.  
   - **In Training Data:** PERCENT, QUANTITY, CARDINAL

8. **LAN** (language)  
   - **Description:** Represents languages.  
   - **In Training Data:** LANGUAGE

## Excluded Entities

The following entities are excluded from our system and will not be mapped:

1. **WEBSITE**
2. **OCC** (occupation)
3. **FAC** (facility: landmarks and places)
4. **PRODUCT**
5. **LAW**
6. **UNIT**
7. **ORDINAL**
